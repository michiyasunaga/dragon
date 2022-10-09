import argparse
import logging
import random
import shutil
import time
import json

# from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import transformers
try:
    from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup
import wandb

from modeling import modeling_dragon
from utils import data_utils
from utils import optimization_utils
from utils import parser_utils
from utils import utils



import numpy as np

import socket, os, sys, subprocess

logger = logging.getLogger(__name__)


def load_data(args, devices, kg):
    _seed = args.seed
    if args.local_rank != -1:
        _seed = args.seed + (2** args.local_rank -1) #use different seed for different gpu process so that they have different training examples
    print ('_seed', _seed, file=sys.stderr)
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(_seed)

    #########################################################
    # Construct the dataset
    #########################################################
    one_process_at_a_time = args.data_loader_one_process_at_a_time

    if args.local_rank != -1 and one_process_at_a_time:
        for p_rank in range(args.world_size):
            if args.local_rank != p_rank: # Barrier
                torch.distributed.barrier()
            dataset = data_utils.DRAGON_DataLoader(args, args.train_statements, args.train_adj,
                args.dev_statements, args.dev_adj,
                args.test_statements, args.test_adj,
                batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                device=devices,
                model_name=args.encoder,
                max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
                is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                subsample=args.subsample, n_train=args.n_train, debug=args.debug, cxt_node_connects_all=args.cxt_node_connects_all, kg=kg)
            if args.local_rank == p_rank: #End of barrier
                torch.distributed.barrier()
    else:
        dataset = data_utils.DRAGON_DataLoader(args, args.train_statements, args.train_adj,
            args.dev_statements, args.dev_adj,
            args.test_statements, args.test_adj,
            batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
            device=devices,
            model_name=args.encoder,
            max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
            is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
            subsample=args.subsample, n_train=args.n_train, debug=args.debug, cxt_node_connects_all=args.cxt_node_connects_all, kg=kg)

    return dataset


def construct_model(args, kg, dataset):
    ########################################################
    #   Load pretrained concept embeddings
    ########################################################
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = np.concatenate(cp_emb, 1)
    cp_emb = torch.tensor(cp_emb, dtype=torch.float)

    concept_num, concept_in_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))
    if args.random_ent_emb:
        cp_emb = None
        freeze_ent_emb = False
        concept_in_dim = args.gnn_dim
    else:
        freeze_ent_emb = args.freeze_ent_emb

    ##########################################################
    #   Build model
    ##########################################################

    if kg == "cpnet":
        n_ntype = 4
        n_etype = 38
        # assert n_etype == dataset.final_num_relation *2
    elif kg == "ddb":
        n_ntype = 4
        n_etype = 34
        # assert n_etype == dataset.final_num_relation *2
    elif kg == "umls":
        n_ntype = 4
        n_etype = dataset.final_num_relation *2
        print ('final_num_relation', dataset.final_num_relation, 'len(id2relation)', len(dataset.id2relation))
        print ('final_num_relation', dataset.final_num_relation, 'len(id2relation)', len(dataset.id2relation), file=sys.stderr)
    else:
        raise ValueError("Invalid KG.")
    if args.cxt_node_connects_all:
        n_etype += 2
    print ('n_ntype', n_ntype, 'n_etype', n_etype)
    print ('n_ntype', n_ntype, 'n_etype', n_etype, file=sys.stderr)
    encoder_load_path = args.encoder_load_path if args.encoder_load_path else args.encoder
    model = modeling_dragon.DRAGON(args, encoder_load_path, k=args.k, n_ntype=n_ntype, n_etype=n_etype, n_concept=concept_num,
        concept_dim=args.gnn_dim,
        concept_in_dim=concept_in_dim,
        n_attention_head=args.att_head_num, fc_dim=args.fc_dim, n_fc_layer=args.fc_layer_num,
        p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf,
        pretrained_concept_emb=cp_emb, freeze_ent_emb=freeze_ent_emb,
        init_range=args.init_range, ie_dim=args.ie_dim, info_exchange=args.info_exchange, ie_layer_num=args.ie_layer_num, sep_ie_layers=args.sep_ie_layers, layer_id=args.encoder_layer)
    return model


def sep_params(model, loaded_roberta_keys):
    """Separate the parameters into loaded and not loaded."""
    loaded_params = dict()
    not_loaded_params = dict()
    params_to_freeze = []
    small_lr_params = dict()
    large_lr_params = dict()
    for n, p in model.named_parameters():
        if n in loaded_roberta_keys:
            loaded_params[n] = p
            params_to_freeze.append(p)
            small_lr_params[n] = p
        else:
            not_loaded_params[n] = p
            large_lr_params[n] = p

    return loaded_params, not_loaded_params, params_to_freeze, small_lr_params, large_lr_params


def count_parameters(loaded_params, not_loaded_params):
    num_params = sum(p.numel() for p in not_loaded_params.values() if p.requires_grad)
    num_fixed_params = sum(p.numel() for p in not_loaded_params.values() if not p.requires_grad)
    num_loaded_params = sum(p.numel() for p in loaded_params.values())
    print('num_trainable_params (out of not_loaded_params):', num_params)
    print('num_fixed_params (out of not_loaded_params):', num_fixed_params)
    print('num_loaded_params:', num_loaded_params)
    print('num_total_params:', num_params + num_fixed_params + num_loaded_params)


def calc_loss_and_acc(logits, labels, loss_type, loss_func):
    if logits is None:
        loss = 0.
        n_corrects = 0
    else:
        if loss_type == 'margin_rank':
            raise NotImplementedError
        elif loss_type == 'cross_entropy':
            loss = loss_func(logits, labels)
        bs = labels.size(0)
        loss *= bs
        n_corrects = (logits.argmax(1) == labels).sum().item()

    return loss, n_corrects


def calc_eval_accuracy(args, eval_set, model, loss_type, loss_func, debug, save_test_preds, preds_path):
    """Eval on the dev or test set - calculate loss and accuracy"""
    total_loss_acm = end_loss_acm = mlm_loss_acm = 0.0
    link_loss_acm = pos_link_loss_acm = neg_link_loss_acm = 0.0
    n_samples_acm = n_corrects_acm = 0
    model.eval()
    save_test_preds = (save_test_preds and args.end_task)
    if save_test_preds:
        utils.check_path(preds_path)
        f_preds = open(preds_path, 'w')
    with torch.no_grad():
        for qids, labels, *input_data in tqdm(eval_set, desc="Dev/Test batch"):
            bs = labels.size(0)
            logits, mlm_loss, link_losses = model(*input_data)
            end_loss, n_corrects = calc_loss_and_acc(logits, labels, loss_type, loss_func)
            link_loss, pos_link_loss, neg_link_loss = link_losses
            loss = args.end_task * end_loss + args.mlm_task * mlm_loss + args.link_task * link_loss

            total_loss_acm += float(loss)
            end_loss_acm += float(end_loss)
            mlm_loss_acm += float(mlm_loss)
            link_loss_acm += float(link_loss)
            pos_link_loss_acm += float(pos_link_loss)
            neg_link_loss_acm += float(neg_link_loss)
            n_corrects_acm += n_corrects
            n_samples_acm += bs

            if save_test_preds:
                predictions = logits.argmax(1) #[bsize, ]
                for qid, pred in zip(qids, predictions):
                    print ('{},{}'.format(qid, chr(ord('A') + pred.item())), file=f_preds)
                    f_preds.flush()
            if debug:
                break
    if save_test_preds:
        f_preds.close()
    total_loss_avg, end_loss_avg, mlm_loss_avg, link_loss_avg, pos_link_loss_avg, neg_link_loss_avg, n_corrects_avg = \
        [item / n_samples_acm for item in (total_loss_acm, end_loss_acm, mlm_loss_acm, link_loss_acm, pos_link_loss_acm, neg_link_loss_acm, n_corrects_acm)]
    return total_loss_avg, end_loss_avg, mlm_loss_avg, link_loss_avg, pos_link_loss_avg, neg_link_loss_avg, n_corrects_avg


def train(args, resume, has_test_split, devices, kg):
    print("args: {}".format(args))

    if resume:
        args.save_dir = os.path.dirname(args.resume_checkpoint)
    if not args.debug:
        if args.local_rank in [-1, 0]:
            log_path = os.path.join(args.save_dir, 'log.csv')
            utils.check_path(log_path)

            # Set up tensorboard
            # tb_dir = os.path.join(args.save_dir, "tb")
            if not resume:
                with open(log_path, 'w') as fout:
                    fout.write('epoch,step,dev_acc,test_acc,best_dev_acc,final_test_acc,best_dev_epoch\n')

                # if os.path.exists(tb_dir):
                #     shutil.rmtree(tb_dir)
            # tb_writer = SummaryWriter(tb_dir)

            config_path = os.path.join(args.save_dir, 'config.json')
            utils.export_config(args, config_path)

    model_path = os.path.join(args.save_dir, 'model.pt')

    dataset = load_data(args, devices, kg)
    dev_dataloader = dataset.dev()
    if has_test_split:
        test_dataloader = dataset.test()

    model = construct_model(args, kg, dataset)
    INHERIT_BERT = os.environ.get('INHERIT_BERT', 0)
    bert_or_roberta = model.lmgnn.bert if INHERIT_BERT else model.lmgnn.roberta
    bert_or_roberta.resize_token_embeddings(len(dataset.tokenizer))

    # Get the names of the loaded LM parameters
    loading_info = model.loading_info
    def _rename_key(key):
        return "lmgnn." + key

    loaded_roberta_keys = [_rename_key(k) for k in loading_info["all_keys"]]

    # Separate the parameters into loaded and not loaded
    loaded_params, not_loaded_params, params_to_freeze, small_lr_params, large_lr_params = sep_params(model, loaded_roberta_keys)

    if args.local_rank in [-1, 0]:
        # print non-loaded parameters
        print(f'Non-loaded parameters: ({len(not_loaded_params.items())} modules)')
        for name, param in not_loaded_params.items():
            if param.requires_grad:
                print('\t{:45}\ttrainable\t{}\tdevice:{}'.format(name, param.size(), param.device))
            else:
                print('\t{:45}\tfixed\t{}\tdevice:{}'.format(name, param.size(), param.device))

        # Count parameters
        count_parameters(loaded_params, not_loaded_params)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    #########################################################
    # Create an optimizer
    #########################################################
    grouped_parameters = [
        {'params': [p for n, p in small_lr_params.items() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in small_lr_params.items() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in large_lr_params.items() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
        {'params': [p for n, p in large_lr_params.items() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
    ]
    optimizer = optimization_utils.OPTIMIZER_CLASSES[args.optim](grouped_parameters)

    #########################################################
    # Optionally loading from a checkpoint
    #########################################################
    if resume:
        print("loading from checkpoint: {}".format(args.resume_checkpoint))
        checkpoint = torch.load(args.resume_checkpoint, map_location='cpu')
        last_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        model.load_state_dict(checkpoint["model"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_dev_epoch = checkpoint["best_dev_epoch"]
        best_dev_acc = checkpoint["best_dev_acc"]
        final_test_acc = checkpoint["final_test_acc"]
        print(f"resume from global_step {global_step}, last_epoch {last_epoch}")
    else:
        last_epoch = -1
        global_step = 0
        best_dev_epoch = best_dev_acc = final_test_acc = 0

    if args.load_model_path and args.load_model_path not in ["None", None]:
        print (f'loading and initializing model from {args.load_model_path}')
        checkpoint = torch.load(args.load_model_path, map_location='cpu')
        model_state_dict = checkpoint["model"]
        try:
            model_state_dict.pop('lmgnn.fc.layers.0-Linear.weight')
            model_state_dict.pop('lmgnn.fc.layers.0-Linear.bias')
        except:
           pass
        model.load_state_dict(model_state_dict, strict=False)


    #########################################################
    # Create a scheduler
    #########################################################
    if args.lr_schedule == 'fixed':
        try:
            scheduler = ConstantLRSchedule(optimizer)
        except:
            scheduler = get_constant_schedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        try:
            scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps, last_epoch=last_epoch)
        except:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, last_epoch=last_epoch)
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
        try:
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=max_steps, last_epoch=last_epoch)
        except:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps, last_epoch=last_epoch)
    if resume:
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("loaded scheduler", checkpoint["scheduler"])

    model.to(devices[1])
    if hasattr(model.lmgnn, 'concept_emb'):
        model.lmgnn.concept_emb.to(devices[0])

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Construct the loss function
    if args.loss == 'margin_rank':
        loss_func = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    elif args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')
    else:
        raise ValueError("Invalid value for args.loss.")

    #############################################################
    #   Training
    #############################################################

    print()
    print('-' * 71)
    if args.fp16:
        print ('Using fp16 training')
        print (f'Upcast {args.upcast}')
        scaler = torch.cuda.amp.GradScaler()

    print ('end_task', args.end_task, 'mlm_task', args.mlm_task, 'link_task', args.link_task)

    total_loss_acm = end_loss_acm = mlm_loss_acm = 0.0
    link_loss_acm = pos_link_loss_acm = neg_link_loss_acm = 0.0
    n_samples_acm = n_corrects_acm = 0
    total_time = 0
    model.train()
    # If all the parameters are frozen in the first few epochs, just skip those epochs.
    if len(params_to_freeze) >= len(list(model.parameters())) - 1:
        args.unfreeze_epoch = 0
    if last_epoch + 1 <= args.unfreeze_epoch:
        utils.freeze_params(params_to_freeze)
    for epoch_id in trange(0, args.n_epochs, desc="Epoch", disable=args.local_rank not in [-1, 0]): #trange(last_epoch + 1, args.n_epochs, desc="Epoch"):
        if last_epoch + 1 > epoch_id:
            time.sleep(1)
            continue
        if epoch_id == args.unfreeze_epoch:
            utils.unfreeze_params(params_to_freeze)
        if epoch_id == args.refreeze_epoch:
            utils.freeze_params(params_to_freeze)
        model.train()

        for qids, labels, *input_data in tqdm(dataset.train(steps=args.redef_epoch_steps, local_rank=args.local_rank), desc="Batch", disable=args.local_rank not in [-1, 0]): #train_dataloader
            # labels: [bs]
            start_time = time.time()
            optimizer.zero_grad()
            bs = labels.size(0)
            a_list = list(range(0, bs, args.mini_batch_size))
            for _idx_, a in enumerate(a_list):
                is_last = (_idx_ == len(a_list) - 1)
                b = min(a + args.mini_batch_size, bs)
                if args.fp16:
                    with torch.cuda.amp.autocast():
                        logits, mlm_loss, link_losses = model(*[x[a:b] for x in input_data]) # logits: [bs, nc]
                        end_loss, n_corrects = calc_loss_and_acc(logits, labels[a:b], args.loss, loss_func)
                else:
                    logits, mlm_loss, link_losses = model(*[x[a:b] for x in input_data]) # logits: [bs, nc]
                    end_loss, n_corrects = calc_loss_and_acc(logits, labels[a:b], args.loss, loss_func)
                link_loss, pos_link_loss, neg_link_loss = link_losses
                loss = args.end_task * end_loss + args.mlm_task * mlm_loss + args.link_task * link_loss

                total_loss_acm += float(loss)
                end_loss_acm += float(end_loss)
                mlm_loss_acm += float(mlm_loss)
                link_loss_acm += float(link_loss)
                pos_link_loss_acm += float(pos_link_loss)
                neg_link_loss_acm += float(neg_link_loss)

                loss = loss / bs
                if (args.local_rank != -1) and (not is_last):
                    with model.no_sync():
                        if args.fp16:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                else:
                    if args.fp16:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                n_corrects_acm += n_corrects
                n_samples_acm += (b - a)

            if args.max_grad_norm > 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scheduler.step()
            # Gradients are accumulated and not back-proped until a batch is processed (not a mini-batch).
            if args.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            total_time += (time.time() - start_time)

            if (global_step + 1) % args.log_interval == 0:
                ms_per_batch = 1000 * total_time / args.log_interval
                if args.local_rank in [-1, 0]:
                    print('| step {:5} |  lr: {:9.7f} | total loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step, scheduler.get_lr()[0], total_loss_acm / n_samples_acm, ms_per_batch))
                    wandb.log({"lr": scheduler.get_lr()[0],
                                "train_loss": total_loss_acm / n_samples_acm,
                                "train_end_loss": end_loss_acm / n_samples_acm,
                                "train_mlm_loss": mlm_loss_acm / n_samples_acm,
                                "train_link_loss": link_loss_acm / n_samples_acm,
                                "train_pos_link_loss": pos_link_loss_acm / n_samples_acm,
                                "train_neg_link_loss": neg_link_loss_acm / n_samples_acm,
                                "train_acc": n_corrects_acm / n_samples_acm,
                                "ms_per_batch": ms_per_batch}, step=global_step)

                total_loss_acm = end_loss_acm = mlm_loss_acm = 0.0
                link_loss_acm = pos_link_loss_acm = neg_link_loss_acm = 0.0
                n_samples_acm = n_corrects_acm = 0
                total_time = 0
            global_step += 1 # Number of batches processed up to now

        # Save checkpoints and evaluate after every epoch
        if args.local_rank in [-1, 0]:
            model.eval()
            preds_path = os.path.join(args.save_dir, 'dev_e{}_preds.csv'.format(epoch_id))
            dev_total_loss, dev_end_loss, dev_mlm_loss, dev_link_loss, dev_pos_link_loss, dev_neg_link_loss, dev_acc = calc_eval_accuracy(args, dev_dataloader, model, args.loss, loss_func, args.debug, not args.debug, preds_path)
            print ('dev_acc', dev_acc)
            if args.end_task and (args.mlm_task or args.link_task):
                dev_dataloader.set_eval_end_task_mode(True)
                _, dev_end_loss, _, _,_,_, dev_acc = calc_eval_accuracy(args, dev_dataloader, model, args.loss, loss_func, args.debug, not args.debug, preds_path)
                dev_dataloader.set_eval_end_task_mode(False)
                print ('dev_acc (eval_end_task_mode)', dev_acc)

            if has_test_split:
                preds_path = os.path.join(args.save_dir, 'test_e{}_preds.csv'.format(epoch_id))
                test_total_loss, test_end_loss, test_mlm_loss, test_link_loss, test_pos_link_loss, test_neg_link_loss, test_acc = calc_eval_accuracy(args, test_dataloader, model, args.loss, loss_func, args.debug, not args.debug, preds_path)
                print ('test_acc', test_acc)
                if args.end_task and (args.mlm_task or args.link_task):
                    test_dataloader.set_eval_end_task_mode(True)
                    _, test_end_loss, _, _,_,_, test_acc = calc_eval_accuracy(args, test_dataloader, model, args.loss, loss_func, args.debug, not args.debug, preds_path)
                    test_dataloader.set_eval_end_task_mode(False)
                    print ('test_acc (eval_end_task_mode)', test_acc)
            else:
                test_acc = 0

            print('-' * 71)
            print('| epoch {:3} | step {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(epoch_id, global_step, dev_acc, test_acc))
            print('-' * 71)

            if dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                final_test_acc = test_acc
                best_dev_epoch = epoch_id
            if not args.debug:
                with open(log_path, 'a') as fout:
                    fout.write('{:3},{:5},{:7.4f},{:7.4f},{:7.4f},{:7.4f},{:3}\n'.format(epoch_id, global_step, dev_acc, test_acc, best_dev_acc, final_test_acc, best_dev_epoch))

            wandb.log({"dev_acc": dev_acc, "dev_loss": dev_total_loss, "dev_end_loss": dev_end_loss, "dev_mlm_loss": dev_mlm_loss, "dev_link_loss": dev_link_loss, "dev_pos_link_loss": dev_pos_link_loss, "dev_neg_link_loss": dev_neg_link_loss, "best_dev_acc": best_dev_acc, "best_dev_epoch": best_dev_epoch}, step=global_step)
            if has_test_split:
                wandb.log({"test_acc": test_acc, "test_loss": test_total_loss, "test_link_loss": test_link_loss, "test_pos_link_loss": test_pos_link_loss, "test_neg_link_loss": test_neg_link_loss, "test_end_loss": test_end_loss, "test_mlm_loss": test_mlm_loss, "final_test_acc": final_test_acc}, step=global_step)
                if args.use_codalab:
                    with open("stats.json", 'w') as fout:
                        json.dump({'epoch': epoch_id, 'step': global_step, 'dev_acc': dev_acc, 'test_acc': test_acc}, fout, indent=2)

            # Save the model checkpoint
            if (args.save_model==2) or ((args.save_model==1) and (best_dev_epoch==epoch_id)):
                if args.local_rank != -1:
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                try:
                    del model_state_dict["lmgnn.concept_emb.emb.weight"]
                except:
                    pass
                checkpoint = {"model": model_state_dict, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch_id, "global_step": global_step, "best_dev_epoch": best_dev_epoch, "best_dev_acc": best_dev_acc, "final_test_acc": final_test_acc, "config": args}
                print('Saving model to {}.{}'.format(model_path, epoch_id))
                torch.save(checkpoint, model_path +".{}".format(epoch_id))

        model.train()
        start_time = time.time()
        if epoch_id > args.unfreeze_epoch and epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
            if args.local_rank in [-1, 0]:
                break

        if args.debug:
            break


def evaluate(args, has_test_split, devices, kg):
    assert args.load_model_path is not None
    load_model_path = args.load_model_path
    print("loading from checkpoint: {}".format(load_model_path))
    checkpoint = torch.load(load_model_path, map_location='cpu')

    train_statements = args.train_statements
    dev_statements = args.dev_statements
    test_statements = args.test_statements
    train_adj = args.train_adj
    dev_adj = args.dev_adj
    test_adj = args.test_adj
    debug = args.debug
    inhouse = args.inhouse

    # args = utils.import_config(checkpoint["config"], args)
    args.train_statements = train_statements
    args.dev_statements = dev_statements
    args.test_statements = test_statements
    args.train_adj = train_adj
    args.dev_adj = dev_adj
    args.test_adj = test_adj
    args.inhouse = inhouse

    dataset = load_data(args, devices, kg)
    dev_dataloader = dataset.dev()
    if has_test_split:
        test_dataloader = dataset.test()
    model = construct_model(args, kg, dataset)
    INHERIT_BERT = os.environ.get('INHERIT_BERT', 0)
    bert_or_roberta = model.lmgnn.bert if INHERIT_BERT else model.lmgnn.roberta
    bert_or_roberta.resize_token_embeddings(len(dataset.tokenizer))

    model.load_state_dict(checkpoint["model"], strict=False)
    epoch_id = checkpoint.get('epoch', 0)

    model.to(devices[1])
    model.lmgnn.concept_emb.to(devices[0])
    model.eval()

    if args.loss == 'margin_rank':
        loss_func = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    elif args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')
    else:
        raise ValueError("Invalid value for args.loss.")

    print ('inhouse?', args.inhouse)

    print ('args.train_statements', args.train_statements)
    print ('args.dev_statements', args.dev_statements)
    print ('args.test_statements', args.test_statements)
    print ('args.train_adj', args.train_adj)
    print ('args.dev_adj', args.dev_adj)
    print ('args.test_adj', args.test_adj)

    model.eval()
    # Evaluation on the dev set
    preds_path = os.path.join(args.save_dir, 'dev_e{}_preds.csv'.format(epoch_id))
    dev_total_loss, dev_end_loss, dev_mlm_loss, dev_link_loss, dev_pos_link_loss, dev_neg_link_loss, dev_acc  = calc_eval_accuracy(args, dev_dataloader, model, args.loss, loss_func, debug, not debug, preds_path)
    if has_test_split:
        # Evaluation on the test set
        preds_path = os.path.join(args.save_dir, 'test_e{}_preds.csv'.format(epoch_id))
        test_total_loss, test_end_loss, test_mlm_loss, test_link_loss, test_pos_link_loss, test_neg_link_loss, test_acc = calc_eval_accuracy(args, test_dataloader, model, args.loss, loss_func, debug, not debug, preds_path)
    else:
        test_acc = 0

    print('-' * 71)
    print('dev_acc {:7.4f}, test_acc {:7.4f}'.format(dev_acc, test_acc))
    print('-' * 71)


def get_devices(args):
    """Get the devices to put the data and the model based on whether to use GPUs and, if so, how many of them are available."""

    if args.local_rank == -1 or not args.cuda:
        if torch.cuda.device_count() >= 2 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:1")
            print("device0: {}, device1: {}".format(device0, device1))
        elif torch.cuda.device_count() == 1 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:0")
        else:
            device0 = torch.device("cpu")
            device1 = torch.device("cpu")
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device0 = torch.device("cuda", args.local_rank)
        device1 = device0
        torch.distributed.init_process_group(backend="nccl")

    args.world_size = world_size = torch.distributed.get_world_size() if args.local_rank != -1 else 1
    print ("Process rank: %s, device: %s, distributed training: %s, world_size: %s" %
              (args.local_rank,
              device0,
              bool(args.local_rank != -1),
              world_size), file=sys.stderr)

    return device0, device1




def main(args):
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(name)s:%(funcName)s():%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.WARNING)

    has_test_split = True
    devices = get_devices(args)

    if not args.use_wandb:
        wandb_mode = "disabled"
    elif args.debug:
        wandb_mode = "offline"
    else:
        wandb_mode = "online"

    # We can optionally resume training from a checkpoint. If doing so, also set the `resume_id` so that you resume your previous wandb run instead of creating a new one.
    resume = args.resume_checkpoint not in [None, "None"]

    args.hf_version = transformers.__version__

    if args.local_rank in [-1, 0]:
        wandb_id = args.resume_id if resume and (args.resume_id not in [None, "None"]) else wandb.util.generate_id()
        args.wandb_id = wandb_id
        wandb.init(project="DRAGON", config=args, name=args.run_name, resume="allow", id=wandb_id, settings=wandb.Settings(start_method="fork"), mode=wandb_mode)
        print(socket.gethostname())
        print ("pid:", os.getpid())
        print ("conda env:", os.environ.get('CONDA_DEFAULT_ENV'))
        print ("screen: %s" % subprocess.check_output('echo $STY', shell=True).decode('utf'))
        print ("gpu: %s" % subprocess.check_output('echo $CUDA_VISIBLE_DEVICES', shell=True).decode('utf'))
        utils.print_cuda_info()
        print("wandb id: ", wandb_id)
        wandb.run.log_code('.')

    kg = args.kg
    if args.dataset == "medqa_usmle":
        kg = "ddb"
    elif args.dataset in ["medqa", "pubmedqa", "bioasq"]:
        kg = "umls"
    print ("KG used:", kg)
    print ("KG used:", kg, file=sys.stderr)

    if args.mode == 'train':
        train(args, resume, has_test_split, devices, kg)
    elif "eval" in args.mode:
        assert args.world_size == 1, "DDP is only implemented for training"
        evaluate(args, has_test_split, devices, kg)
    else:
        raise ValueError('Invalid mode')


if __name__ == '__main__':
    __spec__ = None

    parser = parser_utils.get_parser()
    args, _ = parser.parse_known_args()

    # General
    parser.add_argument('--mode', default='train', choices=['train', 'eval'], help='run training or evaluation')
    parser.add_argument('--use_codalab', default=0, type=int, help='using codalab or not')
    parser.add_argument('--save_dir', default=f'./saved_models/', help='model output directory')
    parser.add_argument('--save_model', default=2, type=float, help="0: do not save model checkpoints. 1: save if best dev. 2: save always")
    parser.add_argument('--load_model_path', default=None, help="The model checkpoint to load in the evaluation mode.")
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    parser.add_argument("--run_name", required=True, type=str, help="The name of this experiment run.")
    parser.add_argument("--resume_checkpoint", default=None, type=str,
                        help="The checkpoint to resume training from.")
    parser.add_argument('--use_wandb', default=False, type=utils.bool_flag, help="Whether to use wandb or not.")
    parser.add_argument("--resume_id", default=None, type=str, help="The wandb run id to resume if `resume_checkpoint` is not None or 'None'.")
    parser.add_argument("--load_graph_cache", default=True, type=utils.bool_flag)
    parser.add_argument("--dump_graph_cache", default=True, type=utils.bool_flag)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--world_size", type=int, default=1, help="For distributed training: world_size")
    parser.add_argument("--data_loader_one_process_at_a_time", default=False, type=utils.bool_flag)


    #Task
    parser.add_argument('--end_task', type=float, default=1.0, help='Task weight for the end task (MCQA)')
    parser.add_argument('--mlm_task', type=float, default=0.0, help='Task weight for the MLM task')
    parser.add_argument('--link_task', type=float, default=0.0, help='Task weight for the LinkPred task')

    parser.add_argument('--mlm_probability', type=float, default=0.15, help='')
    parser.add_argument('--span_mask', type=utils.bool_flag, default=False, help='')
    parser.add_argument('--link_drop_max_count', type=int, default=100, help='To specify #target positive triples for LinkPred')
    parser.add_argument('--link_drop_probability', type=float, default=0.2, help='To specify #target positive triples for LinkPred')
    parser.add_argument('--link_drop_probability_in_which_keep', type=float, default=0.2, help='Within target positive triples, how much to keep in the input graph?')
    parser.add_argument('--link_negative_sample_size', type=int, default=64, help='')
    parser.add_argument('--link_negative_adversarial_sampling', type=utils.bool_flag, default=True, help='')
    parser.add_argument('--link_negative_adversarial_sampling_temperature', type=float, default=1, help='')
    parser.add_argument('--link_regularizer_weight', type=float, default=0.01, help='')
    parser.add_argument('--link_normalize_headtail', type=int, default=0, help='')
    parser.add_argument('--link_proj_headtail', type=utils.bool_flag, default=False, help='')
    parser.add_argument('--scaled_distmult', type=utils.bool_flag, default=False, help='')
    parser.add_argument('--link_gamma', type=float, default=12, help='')
    parser.add_argument('--link_decoder', type=str, default="DistMult", help='')

    # Data
    parser.add_argument('--kg', default='cpnet', help="What KG to use.")
    parser.add_argument('--max_num_relation', default=-1, type=int, help="max number of KG relation types to keep.")
    parser.add_argument('--kg_only_use_qa_nodes', default=False, type=utils.bool_flag, help="")

    parser.add_argument('--train_adj', default=f'{args.data_dir}/{args.dataset}/graph/train.graph.adj.pk', help="The path to the retrieved KG subgraphs of the training set.")
    parser.add_argument('--dev_adj', default=f'{args.data_dir}/{args.dataset}/graph/dev.graph.adj.pk', help="The path to the retrieved KG subgraphs of the dev set.")
    parser.add_argument('--test_adj', default=f'{args.data_dir}/{args.dataset}/graph/test.graph.adj.pk', help="The path to the retrieved KG subgraphs of the test set.")
    parser.add_argument('--max_node_num', default=200, type=int, help="Max number of nodes / the threshold used to prune nodes.")
    parser.add_argument('--subsample', default=1.0, type=float, help="The ratio to subsample the training set.")
    parser.add_argument('--n_train', default=-1, type=int, help="Number of training examples to use. Setting it to -1 means using the `subsample` argument to determine the training set size instead; otherwise it will override the `subsample` argument.")

    # Model architecture
    parser.add_argument('-k', '--k', default=5, type=int, help='The number of Fusion layers')
    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads of the final graph nodes\' pooling')
    parser.add_argument('--gnn_dim', default=100, type=int, help='dimension of the GNN layers')
    parser.add_argument('--fc_dim', default=200, type=int, help='number of FC hidden units (except for the MInt operators)')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of hidden layers of the final MLP')
    parser.add_argument('--freeze_ent_emb', default=True, type=utils.bool_flag, nargs='?', const=True, help='Whether to freeze the entity embedding layer.')
    parser.add_argument('--ie_dim', default=200, type=int, help='number of the hidden units of the MInt operator.')
    parser.add_argument('--residual_ie', default=0, type=int, help='Whether to use residual MInt.')
    parser.add_argument('--info_exchange', default=True, choices=[True, False, "every-other-layer"], type=utils.bool_str_flag, help="Whether we have the MInt operator in every Fusion layer or every other Fusion layer or not at all.")
    parser.add_argument('--ie_layer_num', default=1, type=int, help='number of hidden layers in the MInt operator')
    parser.add_argument("--sep_ie_layers", default=False, type=utils.bool_flag, help="Whether to share parameters across the MInt ops across differernt Fusion layers or not. Setting it to `False` means sharing.")
    parser.add_argument('--random_ent_emb', default=False, type=utils.bool_flag, nargs='?', const=True, help='Whether to use randomly initialized learnable entity embeddings or not.')
    parser.add_argument("--cxt_node_connects_all", default=False, type=utils.bool_flag, help="Whether to connect the interaction node to all the retrieved KG nodes or only the linked nodes.")
    parser.add_argument('--no_node_score', default=True, type=utils.bool_flag, help='Don\'t use node score.')


    # Regularization
    parser.add_argument('--dropouti', type=float, default=0.2, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')

    # Optimization
    parser.add_argument('-dlr', '--decoder_lr', default=1e-3, type=float, help='Learning rate of parameters not in LM')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=2, type=int)
    parser.add_argument('--unfreeze_epoch', default=4, type=int, help="Number of the first few epochs in which LM’s parameters are kept frozen.")
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')
    parser.add_argument('--fp16', default=False, type=utils.bool_flag, help='use fp16 training. this requires torch>=1.6.0')
    parser.add_argument('--upcast', default=False, type=utils.bool_flag, help='Upcast attention computation during fp16 training')
    parser.add_argument('--redef_epoch_steps', default=-1, type=int)

    args = parser.parse_args()
    args.fp16 = args.fp16 and (torch.__version__ >= '1.6.0')
    if args.local_rank != -1:
        assert not args.dump_graph_cache
    main(args)
