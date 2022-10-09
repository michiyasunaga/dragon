import os
import sys
import json
import torch
import random
import pickle
import itertools
import numpy as np

from tqdm import tqdm
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import (OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP)
from transformers import (OpenAIGPTTokenizer, BertTokenizer, BertTokenizerFast, XLNetTokenizer, RobertaTokenizer, RobertaTokenizerFast)
try:
    from transformers import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
    from transformers import AlbertTokenizer
except:
    pass

from preprocess_utils import conceptnet
from utils import utils


MODEL_CLASS_TO_NAME = {
    'gpt': list(OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'bert': list(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'xlnet': list(XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'roberta': list(ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'lstm': ['lstm'],
}
try:
    MODEL_CLASS_TO_NAME['albert'] =  list(ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())
except:
    pass

MODEL_NAME_TO_CLASS = {model_name: model_class for model_class, model_name_list in MODEL_CLASS_TO_NAME.items() for model_name in model_name_list}

#Add SapBERT, PubMedBERT configuration
model_name = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
MODEL_NAME_TO_CLASS[model_name] = 'bert'
model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
MODEL_NAME_TO_CLASS[model_name] = 'bert'
model_name = 'michiyasunaga/BioLinkBERT-large'
MODEL_NAME_TO_CLASS[model_name] = 'bert'


GPT_SPECIAL_TOKENS = ['_start_', '_delimiter_', '_classify_']



class MultiGPUSparseAdjDataBatchGenerator(object):
    """A data generator that batches the data and moves them to the corresponding devices."""
    def __init__(self, args, device0, device1, batch_size, indexes, qids, labels,
                 tensors0=[], lists0=[], tensors1=[], lists1=[], adj_data=None, tokenizer=None):
        self.args = args
        self.device0 = device0
        self.device1 = device1
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors0 = tensors0
        self.lists0 = lists0
        self.tensors1 = tensors1
        self.lists1 = lists1
        self.adj_data = adj_data
        self.tokenizer = tokenizer

        self.mlm_probability = args.mlm_probability
        if args.span_mask:
            print ('span_mask', args.span_mask, file=sys.stderr)
        self.geo_p = 0.2
        self.span_len_upper = 10
        self.span_len_lower = 1
        self.span_lens = list(range(self.span_len_lower, self.span_len_upper + 1))
        self.span_len_dist = [self.geo_p * (1-self.geo_p)**(i - self.span_len_lower) for i in range(self.span_len_lower, self.span_len_upper + 1)]
        self.span_len_dist = [x / (sum(self.span_len_dist)) for x in self.span_len_dist]

        self.eval_end_task_mode = False #if True, use the non-modified text and KG inputs

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        bs = self.batch_size
        n = self.indexes.size(0)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes], self.device1)

            batch_tensors0 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors0]
            assert len(batch_tensors0) == 4 #tensors0: all_input_ids, all_input_mask, all_segment_ids, all_output_mask
            batch_lm_inputs, batch_lm_labels = self.process_lm_data(batch_tensors0)

            batch_tensors1 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors1]
            assert len(batch_tensors1) == 5 #tensors1: concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask
            batch_tensors1[0] = batch_tensors1[0].to(self.device0)
            batch_lists0 = [self._to_device([x[i] for i in batch_indexes], self.device0) for x in self.lists0]
            batch_lists1 = [self._to_device([x[i] for i in batch_indexes], self.device1) for x in self.lists1]

            edge_index_all, edge_type_all = self.adj_data
            #edge_index_all: nested list of shape (n_samples, num_choice), where each entry is tensor[2, E]
            #edge_type_all:  nested list of shape (n_samples, num_choice), where each entry is tensor[E, ]
            edge_index = self._to_device([edge_index_all[i] for i in batch_indexes], self.device1)
            edge_type  = self._to_device([edge_type_all[i] for i in batch_indexes], self.device1)
            node_type_ids = batch_tensors1[1] #[bs, nc, n_nodes]
            assert node_type_ids.dim() == 3
            edge_index, edge_type, pos_triples, neg_nodes = self.process_graph_data(edge_index, edge_type, node_type_ids)

            yield tuple([batch_qids, batch_labels, batch_lm_inputs, batch_lm_labels, *batch_tensors0, *batch_lists0, *batch_tensors1, *batch_lists1, edge_index, edge_type, pos_triples, neg_nodes])

    def _to_device(self, obj, device):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)

    def set_eval_end_task_mode(self, flag: bool):
        self.eval_end_task_mode = flag

    def process_lm_data(self, batch_tensors0):
        input_ids, special_tokens_mask = batch_tensors0[0], batch_tensors0[3]
        assert input_ids.dim() == 3 and special_tokens_mask.dim() == 3
        _bs, _nc, _seqlen = input_ids.size()

        _inputs = input_ids.clone().view(-1, _seqlen) #remember to clone input_ids
        _mask_labels = []
        for ex in _inputs:
            if self.args.span_mask:
                _mask_label = self._span_mask(self.tokenizer.convert_ids_to_tokens(ex))
            else:
                _mask_label = self._word_mask(self.tokenizer.convert_ids_to_tokens(ex))
            _mask_labels.append(_mask_label)
        _mask_labels = torch.tensor(_mask_labels, device=_inputs.device)

        batch_lm_inputs, batch_lm_labels = self.mask_tokens(inputs=_inputs, mask_labels=_mask_labels, special_tokens_mask=special_tokens_mask.view(-1, _seqlen))

        batch_lm_inputs = batch_lm_inputs.view(_bs, _nc, _seqlen) #this is masked
        batch_lm_labels = batch_lm_labels.view(_bs, _nc, _seqlen)

        if self.eval_end_task_mode or (self.args.mlm_task==0):
            batch_lm_inputs = input_ids #non-modified input

        return batch_lm_inputs, batch_lm_labels

    def mask_tokens(self, inputs, mask_labels, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        assert inputs.size() == mask_labels.size()

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        # if self.tokenizer._pad_token is not None: #should be handled already
        #     padding_mask = labels.eq(self.tokenizer.pad_token_id)
        #     probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=labels.device)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=labels.device)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=labels.device)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def _word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masking tokens at word level
        """
        effective_num_toks = 0
        cand_indexes = []
        if isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            after_special_tok = False
            for (i, token) in enumerate(input_tokens):
                if token in ["[CLS]", "[SEP]", "[PAD]"]:
                    after_special_tok = True
                    continue
                if len(cand_indexes) >= 1 and (not after_special_tok) and token.startswith("##"):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])
                after_special_tok = False
                effective_num_toks += 1
        elif isinstance(self.tokenizer, (RobertaTokenizer, RobertaTokenizerFast)):
            after_special_tok = False
            for (i, token) in enumerate(input_tokens):
                if token in ["<s>",  "</s>", "<pad>"]:
                    after_special_tok = True
                    continue
                if len(cand_indexes) >= 1 and (not after_special_tok) and (not token.startswith("Ġ")):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])
                after_special_tok = False
                effective_num_toks += 1
        else:
            raise NotImplementedError

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(effective_num_toks * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def _span_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masking tokens at word level
        """
        effective_num_toks = 0
        cand_indexes = []
        if isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            after_special_tok = False
            for (i, token) in enumerate(input_tokens):
                if token in ["[CLS]", "[SEP]", "[PAD]"]:
                    after_special_tok = True
                    continue
                if len(cand_indexes) >= 1 and (not after_special_tok) and token.startswith("##"):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])
                after_special_tok = False
                effective_num_toks += 1
        elif isinstance(self.tokenizer, (RobertaTokenizer, RobertaTokenizerFast)):
            after_special_tok = False
            for (i, token) in enumerate(input_tokens):
                if token in ["<s>",  "</s>", "<pad>"]:
                    after_special_tok = True
                    continue
                if len(cand_indexes) >= 1 and (not after_special_tok) and (not token.startswith("Ġ")):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])
                after_special_tok = False
                effective_num_toks += 1
        else:
            raise NotImplementedError
        cand_indexes_args = list(range(len(cand_indexes)))

        random.shuffle(cand_indexes_args)
        num_to_predict = min(max_predictions, max(1, int(round(effective_num_toks * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for wid in cand_indexes_args:
            if len(masked_lms) >= num_to_predict:
                break
            span_len = np.random.choice(self.span_lens, p=self.span_len_dist)
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            # if len(masked_lms) + span_len > num_to_predict:
            #     continue
            index_set = []
            is_any_index_covered = False
            for _wid in range(wid, len(cand_indexes)): #iterate over word
                if len(index_set) + len(cand_indexes[_wid]) > span_len:
                    break
                for _index in cand_indexes[_wid]: #iterate over subword
                    if _index in covered_indexes:
                        is_any_index_covered = True
                        break
                    index_set.append(_index)
                if is_any_index_covered:
                    break
            if is_any_index_covered:
                continue
            for _index in index_set:
                covered_indexes.add(_index)
                masked_lms.append(_index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def process_graph_data(self, edge_index, edge_type, node_type_ids):
        #edge_index: nested list of shape (n_samples, num_choice), where each entry is tensor[2, E]
        #edge_type:  nested list of shape (n_samples, num_choice), where each entry is tensor[E, ]
        #node_type_ids: tensor[n_samples, num_choice, num_nodes]
        bs, nc = len(edge_index), len(edge_index[0])
        input_edge_index, input_edge_type, pos_triples, neg_nodes = [], [], [], []
        for bid in range(bs):
            for cid in range(nc):
                _edge_index = edge_index[bid][cid] #.clone()
                _edge_type  = edge_type[bid][cid] #.clone()
                _node_type_ids = node_type_ids[bid][cid] #.clone()
                _edge_index, _edge_type, _pos_triples, _neg_nodes = self._process_one_graph(_edge_index, _edge_type, _node_type_ids)
                input_edge_index.append(_edge_index)
                input_edge_type.append(_edge_type)
                pos_triples.append(_pos_triples)
                neg_nodes.append(_neg_nodes)
        input_edge_index = list(map(list, zip(*(iter(input_edge_index),) * nc))) #nested list of shape (n_samples, num_choice)
        input_edge_type  = list(map(list, zip(*(iter(input_edge_type),) * nc)))
        pos_triples = list(map(list, zip(*(iter(pos_triples),) * nc)))
        neg_nodes   = list(map(list, zip(*(iter(neg_nodes),) * nc)))

        if self.eval_end_task_mode or (self.args.link_task==0):
            input_edge_index = edge_index  #non-modified input
            input_edge_type = edge_type    #non-modified input

        return input_edge_index, input_edge_type, pos_triples, neg_nodes

    def _process_one_graph(self, _edge_index, _edge_type, _node_type_ids):
        #_edge_index: tensor[2, E]
        #_edge_type:  tensor[E, ]
        #_node_type_ids: tensor[n_nodes, ]
        E = len(_edge_type)
        if E == 0:
            # print ('KG with 0 node', file=sys.stderr)
            effective_num_nodes = 1
        else:
            effective_num_nodes = int(_edge_index.max()) + 1
        device = _edge_type.device

        tmp = _node_type_ids.max().item()
        assert isinstance(tmp, int) and 0 <= tmp <= 5
        _edge_index_node_type = _node_type_ids[_edge_index] #[2, E]
        _is_special = (_edge_index_node_type == 3) #[2, E]
        is_special = _is_special[0] | _is_special[1] #[E,]

        positions = torch.arange(E)
        positions = positions[~is_special] #[some_E, ]
        drop_count = min(self.args.link_drop_max_count, int(len(positions) * self.args.link_drop_probability))
        if len(positions) > 0 and drop_count > 0:
            drop_idxs = torch.multinomial(torch.full((len(positions),), 1.0), drop_count, replacement=False) #[drop_count, ]
        else:
            drop_idxs = torch.tensor([]).long()
        drop_positions = positions[drop_idxs] #[drop_count, ]

        mask = torch.zeros((E,)).long() #[E, ]
        mask = mask.index_fill_(dim=0, index=drop_positions, value=1).bool().to(device) #[E, ]

        real_drop_count = int(drop_count * (1-self.args.link_drop_probability_in_which_keep))
        real_drop_positions = positions[drop_idxs[:real_drop_count]] #[real_drop_count, ]
        real_mask = torch.zeros((E,)).long() #[E, ]
        real_mask = real_mask.index_fill_(dim=0, index=real_drop_positions, value=1).bool().to(device) #[E, ]


        assert int(mask.long().sum()) == drop_count
        # print (f'drop_E / total_E = {drop_count} / {E} = {drop_count / E}', ) #E is typically 1000-3000
        input_edge_index = _edge_index[:, ~real_mask]
        input_edge_type  = _edge_type[~real_mask]
        assert input_edge_index.size(1) == E - real_drop_count

        pos_edge_index = _edge_index[:, mask]
        pos_edge_type  = _edge_type[mask]
        pos_triples = [pos_edge_index[0], pos_edge_type, pos_edge_index[1]]
        #pos_triples: list[h, r, t], where each of h, r, t is [n_triple, ]
        assert pos_edge_index.size(1) == drop_count

        num_edges = len(pos_edge_type)
        num_corruption = self.args.link_negative_sample_size
        neg_nodes = torch.randint(0, effective_num_nodes, (num_edges, num_corruption), device=device) #[n_triple, n_neg]
        return input_edge_index, input_edge_type, pos_triples, neg_nodes


class DRAGON_DataLoader(object):

    def __init__(self, args, train_statement_path, train_adj_path,
                 dev_statement_path, dev_adj_path,
                 test_statement_path, test_adj_path,
                 batch_size, eval_batch_size, device, model_name, max_node_num=200, max_seq_length=128,
                 is_inhouse=False, inhouse_train_qids_path=None,
                 subsample=1.0, n_train=-1, debug=False, cxt_node_connects_all=False, kg="cpnet"):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device0, self.device1 = device
        self.is_inhouse = is_inhouse
        self.debug = debug
        self.model_name = model_name
        self.max_node_num = max_node_num
        self.debug_sample_size = 32
        self.cxt_node_connects_all = cxt_node_connects_all

        self.model_type = MODEL_NAME_TO_CLASS[model_name]
        self.load_resources(kg)

        # Load training data
        print ('train_statement_path', train_statement_path)
        self.train_qids, self.train_labels, self.train_encoder_data, train_concepts_by_sents_list = self.load_input_tensors(train_statement_path, max_seq_length, mode='train')

        num_choice = self.train_encoder_data[0].size(1)
        self.num_choice = num_choice
        print ('num_choice', num_choice)
        *self.train_decoder_data, self.train_adj_data = self.load_sparse_adj_data_with_contextnode(train_adj_path, max_node_num, train_concepts_by_sents_list, mode='train')
        if not debug:
            assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)

        print("Finish loading training data.")

        # Load dev data
        self.dev_qids, self.dev_labels, self.dev_encoder_data, dev_concepts_by_sents_list = self.load_input_tensors(dev_statement_path, max_seq_length)
        *self.dev_decoder_data, self.dev_adj_data = self.load_sparse_adj_data_with_contextnode(dev_adj_path, max_node_num, dev_concepts_by_sents_list)
        if not debug:
            assert all(len(self.dev_qids) == len(self.dev_adj_data[0]) == x.size(0) for x in [self.dev_labels] + self.dev_encoder_data + self.dev_decoder_data)

        print("Finish loading dev data.")

        # Load test data
        if test_statement_path is not None:
            self.test_qids, self.test_labels, self.test_encoder_data, test_concepts_by_sents_list = self.load_input_tensors(test_statement_path, max_seq_length)
            *self.test_decoder_data, self.test_adj_data = self.load_sparse_adj_data_with_contextnode(test_adj_path, max_node_num, test_concepts_by_sents_list)
            if not debug:
                assert all(len(self.test_qids) == len(self.test_adj_data[0]) == x.size(0) for x in [self.test_labels] + self.test_encoder_data + self.test_decoder_data)

            print("Finish loading test data.")

        # If using inhouse split, we split the original training set into an inhouse training set and an inhouse test set.
        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

        # Optionally we can subsample the training set.
        assert 0. < subsample <= 1.
        if subsample < 1. or n_train >= 0:
            # n_train will override subsample if the former is not None
            if n_train == -1:
                n_train = int(self.train_size() * subsample)
            assert n_train > 0
            if self.is_inhouse:
                self.inhouse_train_indexes = self.inhouse_train_indexes[:n_train]
            else:
                self.train_qids = self.train_qids[:n_train]
                self.train_labels = self.train_labels[:n_train]
                self.train_encoder_data = [x[:n_train] for x in self.train_encoder_data]
                self.train_decoder_data = [x[:n_train] for x in self.train_decoder_data]
                self.train_adj_data = tuple([x[:n_train] for x in self.train_adj_data])
                assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
            assert self.train_size() == n_train

    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        else:
            return len(self.test_qids) if hasattr(self, 'test_qids') else 0

    def train(self, steps=-1, local_rank=-1):
        if self.debug:
            train_indexes = torch.arange(self.debug_sample_size)
        elif self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        if steps != -1:
            train_indexes = train_indexes[: self.args.batch_size * steps]

        print ('local_rank', local_rank, 'len(train_indexes)', len(train_indexes), 'train_indexes[:10]', train_indexes[:10].tolist())
        print ('local_rank', local_rank, 'len(train_indexes)', len(train_indexes), 'train_indexes[:10]', train_indexes[:10].tolist(), file=sys.stderr)
        return MultiGPUSparseAdjDataBatchGenerator(self.args, self.device0, self.device1, self.batch_size, train_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data, tokenizer=self.tokenizer)

    def train_eval(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.train_qids)), self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data, tokenizer=self.tokenizer)

    def dev(self):
        if self.debug:
            dev_indexes = torch.arange(self.debug_sample_size)
        else:
            dev_indexes = torch.arange(len(self.dev_qids))
        return MultiGPUSparseAdjDataBatchGenerator(self.args, self.device0, self.device1, self.eval_batch_size, dev_indexes, self.dev_qids, self.dev_labels, tensors0=self.dev_encoder_data, tensors1=self.dev_decoder_data, adj_data=self.dev_adj_data, tokenizer=self.tokenizer)

    def test(self):
        if self.debug:
            test_indexes = torch.arange(self.debug_sample_size)
        elif self.is_inhouse:
            test_indexes = self.inhouse_test_indexes
        else:
            test_indexes = torch.arange(len(self.test_qids))
        if self.is_inhouse:
            return MultiGPUSparseAdjDataBatchGenerator(self.args, self.device0, self.device1, self.eval_batch_size, test_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data, tokenizer=self.tokenizer)
        else:
            return MultiGPUSparseAdjDataBatchGenerator(self.args, self.device0, self.device1, self.eval_batch_size, test_indexes, self.test_qids, self.test_labels, tensors0=self.test_encoder_data, tensors1=self.test_decoder_data, adj_data=self.test_adj_data, tokenizer=self.tokenizer)

    def load_resources(self, kg):
        # Load the tokenizer
        try:
            tokenizer_class = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer, 'roberta': RobertaTokenizer, 'albert': AlbertTokenizer}.get(self.model_type)
        except:
            tokenizer_class = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer, 'roberta': RobertaTokenizer}.get(self.model_type)
        tokenizer = tokenizer_class.from_pretrained(self.model_name)
        self.tokenizer = tokenizer

        if kg == "cpnet":
            # Load cpnet
            cpnet_vocab_path = self.args.kg_vocab_path #"data/cpnet/concept.txt"
            with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
                self.id2concept = [w.strip() for w in fin]
            self.concept2id = {w: i for i, w in enumerate(self.id2concept)}
            self.id2relation = conceptnet.merged_relations
        elif kg == "ddb":
            cpnet_vocab_path = self.args.kg_vocab_path #"data/ddb/vocab.txt"
            with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
                self.id2concept = [w.strip() for w in fin]
            self.concept2id = {w: i for i, w in enumerate(self.id2concept)}
            self.id2relation = [
                'belongstothecategoryof',
                'isacategory',
                'maycause',
                'isasubtypeof',
                'isariskfactorof',
                'isassociatedwith',
                'maycontraindicate',
                'interactswith',
                'belongstothedrugfamilyof',
                'child-parent',
                'isavectorfor',
                'mabeallelicwith',
                'seealso',
                'isaningradientof',
                'mabeindicatedby'
            ]
        elif kg == "umls":
            cpnet_vocab_path = self.args.kg_vocab_path #"data/umls/concepts.txt"
            with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
                self.id2concept = [w.strip() for w in fin]
            self.concept2id = {w: i for i, w in enumerate(self.id2concept)}
            self.id2relation = [r.strip() for r in open(f"{os.path.dirname(self.args.kg_vocab_path)}/relations.txt")]
        else:
            raise ValueError("Invalid value for kg.")

    def load_input_tensors(self, input_jsonl_path, max_seq_length, mode='eval'):
        """Construct input tensors for the LM component of the model."""
        cache_path = input_jsonl_path + "-sl{}".format(max_seq_length) + (("-" + self.model_type) if self.model_type != "roberta" else "") + '.loaded_cache'
        use_cache = True

        if use_cache and not os.path.exists(cache_path):
            use_cache = False

        if use_cache:
            print (f'Loading cache {cache_path}')
            print (f'Loading cache {cache_path}', file=sys.stderr)
            # with open(cache_path, 'rb') as f:
            #     input_tensors = utils.CPU_Unpickler(f).load()
            input_tensors = ()
            with open(cache_path, "rb") as in_file:
                try:
                    while True:
                        obj = pickle.load(in_file)
                        if type(obj) == dict:
                            assert len(obj) == 1
                            key = list(obj.keys())[0]
                            input_tensors = input_tensors + (obj[key], )
                        elif type(obj) == tuple:
                            assert len(obj) == 4 #example_ids, all_label, data_tensors, concepts_by_sents_list
                            input_tensors = obj
                        else:
                            raise TypeError("Invalid type for obj.")
                except EOFError:
                    pass
            print (f'Loaded cache {cache_path}', file=sys.stderr)
        else:
            if self.model_type in ('lstm',):
                raise NotImplementedError
            elif self.model_type in ('gpt',):
                input_tensors = load_gpt_input_tensors(input_jsonl_path, max_seq_length)
            elif self.model_type in ('bert', 'xlnet', 'roberta', 'albert'):
                input_tensors = load_bert_xlnet_roberta_input_tensors(input_jsonl_path, max_seq_length, self.debug, self.tokenizer, self.debug_sample_size)
            if not self.debug:
                if self.args.local_rank in [-1, 0]:
                    print ('saving cache...', file=sys.stderr)
                    # utils.save_pickle(input_tensors, cache_path)
                    with open(cache_path, 'wb') as f:
                        for _i_, obj in enumerate(tqdm(input_tensors)):
                            pickle.dump({f'obj{_i_}': obj}, f, protocol=4)
                    print ('saved cache.', file=sys.stderr)

        if mode == 'train' and self.args.local_rank != -1:
            example_ids, all_label, data_tensors, concepts_by_sents_list = input_tensors #concepts_by_sents_list is always []
            assert len(example_ids) == len(all_label) == len(data_tensors[0])
            total_num = len(data_tensors[0])
            rem = total_num % self.args.world_size
            if rem != 0:
                example_ids = example_ids + example_ids[:self.args.world_size - rem]
                all_label = torch.cat([all_label, all_label[:self.args.world_size - rem]], dim=0)
                data_tensors = [torch.cat([t, t[:self.args.world_size - rem]], dim=0) for t in data_tensors]
                total_num_aim = total_num + self.args.world_size - rem
            else:
                total_num_aim = total_num
            assert total_num_aim % self.args.world_size == 0
            assert total_num_aim == len(data_tensors[0])
            _select = (torch.arange(total_num_aim) % self.args.world_size) == self.args.local_rank #bool tensor
            example_ids = np.array(example_ids)[_select].tolist()
            all_label = all_label[_select]
            data_tensors = [t[_select] for t in data_tensors]
            input_tensors = (example_ids, all_label, data_tensors, [])
        example_ids = input_tensors[0]
        print ('local_rank', self.args.local_rank, 'len(example_ids)', len(example_ids), file=sys.stderr)
        return input_tensors

    def load_sparse_adj_data_with_contextnode(self, adj_pk_path, max_node_num, concepts_by_sents_list, mode='eval'):
        """Construct input tensors for the GNN component of the model."""
        print("Loading sparse adj data...")
        cache_path = adj_pk_path + "-nodenum{}".format(max_node_num) + ("-cntsall" if self.cxt_node_connects_all else "") + '.loaded_cache'
        # use_cache = self.args.dump_graph_cache
        use_cache = self.args.load_graph_cache

        if use_cache and not os.path.exists(cache_path):
            use_cache = False

        if use_cache:
            print (f'Loading cache {cache_path}')
            print (f'Loading cache {cache_path}', file=sys.stderr)
            # with open(cache_path, 'rb') as f:
            #     adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel, special_nodes_mask = utils.CPU_Unpickler(f).load()
            loaded_data = []
            with open(cache_path, "rb") as in_file:
                try:
                    while True:
                        obj = pickle.load(in_file)
                        if type(obj) == dict:
                            assert len(obj) == 1
                            key = list(obj.keys())[0]
                            loaded_data.append(obj[key])
                        elif type(obj) == list:
                            loaded_data.extend(obj)
                        else:
                            raise TypeError("Invalid type for obj.")
                except EOFError:
                    pass
            adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel, special_nodes_mask = loaded_data
            self.final_num_relation = half_n_rel
            print (f'Loaded cache {cache_path}', file=sys.stderr)

            ori_adj_mean  = adj_lengths_ori.float().mean().item()
            ori_adj_sigma = np.sqrt(((adj_lengths_ori.float() - ori_adj_mean)**2).mean().item())
            print('| ori_adj_len: mu {:.2f} sigma {:.2f} | adj_len: {:.2f} |'.format(ori_adj_mean, ori_adj_sigma, adj_lengths.float().mean().item()) +
                ' prune_rate： {:.2f} |'.format((adj_lengths_ori > adj_lengths).float().mean().item()) +
                ' qc_num: {:.2f} | ac_num: {:.2f} |'.format((node_type_ids == 0).float().sum(1).mean().item(),
                                                            (node_type_ids == 1).float().sum(1).mean().item()))

            edge_index = list(map(list, zip(*(iter(edge_index),) * self.num_choice))) #list of size (n_questions, n_choices), where each entry is tensor[2, E] #this operation corresponds to .view(n_questions, n_choices)
            edge_type = list(map(list, zip(*(iter(edge_type),) * self.num_choice))) #list of size (n_questions, n_choices), where each entry is tensor[E, ]

            adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask = [x.view(-1, self.num_choice, *x.size()[1:]) for x in (adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask)]
            #concept_ids: (n_questions, num_choice, max_node_num)
            #node_type_ids: (n_questions, num_choice, max_node_num)
            #node_scores: (n_questions, num_choice, max_node_num)
            #adj_lengths: (n_questions,　num_choice)

            if mode == 'train' and self.args.local_rank != -1:
                assert len(adj_lengths_ori) == len(concept_ids) == len(node_type_ids) == len(node_scores) == len(adj_lengths) == len(edge_index) == len(edge_type) == len(special_nodes_mask) #they equal to n_questions * num_choice
                _ts = [adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask]
                total_num = len(edge_index)
                rem = total_num % self.args.world_size
                if rem != 0:
                    edge_index = edge_index + edge_index[:self.args.world_size - rem]
                    edge_type  = edge_type + edge_type[:self.args.world_size - rem]
                    _ts = [torch.cat([t, t[:self.args.world_size - rem]], dim=0) for t in _ts]
                    total_num_aim = total_num + self.args.world_size - rem
                else:
                    total_num_aim = total_num
                assert total_num_aim % self.args.world_size == 0
                assert total_num_aim == len(_ts[0]) == len(edge_index)
                _select = (torch.arange(total_num_aim) % self.args.world_size) == self.args.local_rank #bool tensor
                edge_index = [e for e, TF in zip(edge_index, _select) if TF]
                edge_type  = [e for e, TF in zip(edge_type, _select) if TF]
                _ts = [t[_select] for t in _ts]
                adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask = _ts
                assert len(adj_lengths_ori) == len(concept_ids) == len(node_type_ids) == len(node_scores) == len(adj_lengths) == len(edge_index) == len(edge_type) == len(special_nodes_mask)
            print ('local_rank', self.args.local_rank, 'len(edge_index)', len(edge_index), file=sys.stderr)
        else:
            # Set special nodes and links
            context_node = 0
            n_special_nodes = 1
            cxt2qlinked_rel = 0
            cxt2alinked_rel = 1
            half_n_rel = len(self.id2relation) + 2
            if self.cxt_node_connects_all:
                cxt2other_rel = half_n_rel
                half_n_rel += 1

            adj_concept_pairs = []
            print(f'Loading {adj_pk_path}...', file=sys.stderr)
            with open(adj_pk_path, "rb") as in_file:
                try:
                    while True:
                        ex = pickle.load(in_file)
                        if type(ex) == dict:
                            adj_concept_pairs.append(ex)
                        elif type(ex) == tuple:
                            adj_concept_pairs.append(ex)
                        elif type(ex) == list:
                            assert len(ex) > 10
                            adj_concept_pairs.extend(ex)
                        else:
                            raise TypeError("Invalid type for ex.")
                except EOFError:
                    pass
            print(f'Loaded {adj_pk_path}...', file=sys.stderr)

            n_samples = len(adj_concept_pairs) #this is actually n_questions x n_choices
            assert n_samples % self.num_choice == 0
            n_questions = n_samples // self.num_choice
            if mode == 'train' and self.args.local_rank != -1:
                rem = n_questions % self.args.world_size
                if rem != 0:
                    adj_concept_pairs = adj_concept_pairs + adj_concept_pairs[: (self.args.world_size - rem) * self.num_choice]
                    n_questions_aim = n_questions + self.args.world_size - rem
                else:
                    n_questions_aim = n_questions
                assert n_questions_aim % self.args.world_size == 0
                n_samples = n_questions_aim // self.args.world_size  * self.num_choice

            edge_index, edge_type = [], []
            adj_lengths = torch.zeros((n_samples,), dtype=torch.long)
            concept_ids = torch.full((n_samples, max_node_num), 1, dtype=torch.long)
            node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long) #default 2: "other node"
            node_scores = torch.zeros((n_samples, max_node_num, 1), dtype=torch.float)
            special_nodes_mask = torch.zeros(n_samples, max_node_num, dtype=torch.bool)

            adj_lengths_ori = adj_lengths.clone()
            if not concepts_by_sents_list:
                concepts_by_sents_list = itertools.repeat(None)
            idx = -1
            for _idx, (_data, cpts_by_sents) in tqdm(enumerate(zip(adj_concept_pairs, concepts_by_sents_list)), total=n_questions * self.num_choice, desc='loading adj matrices'):
                if self.debug and _idx >= self.debug_sample_size * self.num_choice:
                    break

                if mode == 'train' and self.args.local_rank != -1:
                    qidx = _idx // self.num_choice
                    if qidx % self.args.world_size != self.args.local_rank:
                        continue
                idx += 1

                if isinstance(_data, dict):
                    adj, concepts, qm, am, cid2score = _data['adj'], _data['concepts'], _data['qmask'], _data['amask'], _data['cid2score']
                else:
                    adj, concepts, qm, am = _data
                    cid2score = None
                #adj: e.g. <4233x249 (n_nodes*half_n_rels x n_nodes) sparse matrix of type '<class 'numpy.bool'>' with 2905 stored elements in COOrdinate format>
                #concepts: np.array(num_nodes, ), where entry is concept id
                #qm: np.array(num_nodes, ), where entry is True/False
                #am: np.array(num_nodes, ), where entry is True/False
                assert len(concepts) == len(set(concepts))
                qam = qm | am
                #sanity check: should be T,..,T,F,F,..F
                if len(concepts) == 0:
                    # print ("KG with 0 node", file=sys.stderr)
                    pass
                else:
                    assert qam[0] == True
                F_start = False
                for TF in qam:
                    if TF == False:
                        F_start = True
                    else:
                        assert F_start == False

                assert n_special_nodes <= max_node_num
                special_nodes_mask[idx, :n_special_nodes] = 1

                if self.args.kg_only_use_qa_nodes:
                    actual_max_node_num = torch.tensor(qam).long().sum().item()
                else:
                    actual_max_node_num = max_node_num
                num_concept = min(len(concepts) + n_special_nodes, actual_max_node_num) #this is the final number of nodes including contextnode but excluding PAD
                adj_lengths_ori[idx] = len(concepts)
                adj_lengths[idx] = num_concept

                #Prepare nodes
                concepts = concepts[:num_concept - n_special_nodes]
                concept_ids[idx, n_special_nodes:num_concept] = torch.tensor(concepts + 1)  #To accomodate contextnode, original concept_ids incremented by 1
                concept_ids[idx, 0] = context_node #this is the "concept_id" for contextnode

                #Prepare node scores
                if cid2score is not None:
                    if -1 not in cid2score:
                        cid2score[-1] = 0
                    for _j_ in range(num_concept):
                        _cid = int(concept_ids[idx, _j_]) - 1 # Now context node is -1
                        node_scores[idx, _j_, 0] = torch.tensor(cid2score[_cid])

                #Prepare node types
                node_type_ids[idx, 0] = 3 # context node
                node_type_ids[idx, 1:n_special_nodes] = 4 # sent nodes
                node_type_ids[idx, n_special_nodes:num_concept][torch.tensor(qm, dtype=torch.bool)[:num_concept - n_special_nodes]] = 0
                node_type_ids[idx, n_special_nodes:num_concept][torch.tensor(am, dtype=torch.bool)[:num_concept - n_special_nodes]] = 1

                #Load adj
                ij = torch.tensor(adj.row, dtype=torch.int64) #(num_matrix_entries, ), where each entry is coordinate
                k = torch.tensor(adj.col, dtype=torch.int64)  #(num_matrix_entries, ), where each entry is coordinate
                n_node = adj.shape[1]
                if n_node > 0:
                    assert len(self.id2relation) == adj.shape[0] // n_node
                    i, j = ij // n_node, ij % n_node
                else:
                    i, j = ij, ij

                #Prepare edges
                i += 2; j += 1; k += 1  # **** increment coordinate by 1, rel_id by 2 ****
                extra_i, extra_j, extra_k = [], [], []
                for _coord, q_tf in enumerate(qm):
                    _new_coord = _coord + n_special_nodes
                    if _new_coord > num_concept:
                        break
                    if q_tf:
                        extra_i.append(cxt2qlinked_rel) #rel from contextnode to question concept
                        extra_j.append(0) #contextnode coordinate
                        extra_k.append(_new_coord) #question concept coordinate
                    elif self.cxt_node_connects_all:
                        extra_i.append(cxt2other_rel) #rel from contextnode to other concept
                        extra_j.append(0) #contextnode coordinate
                        extra_k.append(_new_coord) #other concept coordinate
                for _coord, a_tf in enumerate(am):
                    _new_coord = _coord + n_special_nodes
                    if _new_coord > num_concept:
                        break
                    if a_tf:
                        extra_i.append(cxt2alinked_rel) #rel from contextnode to answer concept
                        extra_j.append(0) #contextnode coordinate
                        extra_k.append(_new_coord) #answer concept coordinate
                    elif self.cxt_node_connects_all:
                        extra_i.append(cxt2other_rel) #rel from contextnode to other concept
                        extra_j.append(0) #contextnode coordinate
                        extra_k.append(_new_coord) #other concept coordinate

                # half_n_rel += 2 #should be 19 now
                if len(extra_i) > 0:
                    i = torch.cat([i, torch.tensor(extra_i)], dim=0)
                    j = torch.cat([j, torch.tensor(extra_j)], dim=0)
                    k = torch.cat([k, torch.tensor(extra_k)], dim=0)

                if self.args.max_num_relation > 0:
                    _keep = (i < self.args.max_num_relation).bool()
                    assert _keep.dim() == 1 and _keep.size(0) == i.size(0) == j.size(0) == k.size(0)
                    i = i[_keep]
                    j = j[_keep]
                    k = k[_keep]
                    half_n_rel = min(half_n_rel, self.args.max_num_relation)
                self.final_num_relation = half_n_rel
                ########################

                mask = (j < actual_max_node_num) & (k < actual_max_node_num)
                i, j, k = i[mask], j[mask], k[mask]
                i, j, k = torch.cat((i, i + half_n_rel), 0), torch.cat((j, k), 0), torch.cat((k, j), 0)  # add inverse relations
                edge_index.append(torch.stack([j,k], dim=0)) #each entry is [2, E]
                edge_type.append(i) #each entry is [E, ]

            if not self.debug and self.args.dump_graph_cache:
                print ('saving cache...', file=sys.stderr)
                with open(cache_path, 'wb') as f:
                    # pickle.dump([adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel, special_nodes_mask], f)
                    for _i_, obj in enumerate(tqdm([adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel, special_nodes_mask])):
                        pickle.dump({f'obj{_i_}': obj}, f, protocol=4)
                print ('saved cache.', file=sys.stderr)

            assert n_samples == idx+1
            print ('local_rank', self.args.local_rank, 'graph loading final idx+1', idx+1, file=sys.stderr)
            del adj_concept_pairs

            ori_adj_mean  = adj_lengths_ori.float().mean().item()
            ori_adj_sigma = np.sqrt(((adj_lengths_ori.float() - ori_adj_mean)**2).mean().item())
            print('| ori_adj_len: mu {:.2f} sigma {:.2f} | adj_len: {:.2f} |'.format(ori_adj_mean, ori_adj_sigma, adj_lengths.float().mean().item()) +
                ' prune_rate： {:.2f} |'.format((adj_lengths_ori > adj_lengths).float().mean().item()) +
                ' qc_num: {:.2f} | ac_num: {:.2f} |'.format((node_type_ids == 0).float().sum(1).mean().item(),
                                                            (node_type_ids == 1).float().sum(1).mean().item()))

            edge_index = list(map(list, zip(*(iter(edge_index),) * self.num_choice))) #list of size (n_questions, n_choices), where each entry is tensor[2, E] #this operation corresponds to .view(n_questions, n_choices)
            edge_type = list(map(list, zip(*(iter(edge_type),) * self.num_choice))) #list of size (n_questions, n_choices), where each entry is tensor[E, ]

            adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask = [x.view(-1, self.num_choice, *x.size()[1:]) for x in (adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask)]
            #concept_ids: (n_questions, num_choice, max_node_num)
            #node_type_ids: (n_questions, num_choice, max_node_num)
            #node_scores: (n_questions, num_choice, max_node_num)
            #adj_lengths: (n_questions,　num_choice)
        return concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask, (edge_index, edge_type) #, half_n_rel * 2 + 1


######################### GPT loader utils #########################
def load_gpt_input_tensors(statement_jsonl_path, max_seq_length):
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def load_qa_dataset(dataset_path):
        """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
        with open(dataset_path, "r", encoding="utf-8") as fin:
            output = []
            for line in fin:
                input_json = json.loads(line)
                label = ord(input_json.get("answerKey", "A")) - ord("A")
                output.append((input_json['id'], input_json["question"]["stem"], *[ending["text"] for ending in input_json["question"]["choices"]], label))
        return output

    def pre_process_datasets(encoded_datasets, num_choices, max_seq_length, start_token, delimiter_token, clf_token):
        """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

            To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
            input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
        """
        tensor_datasets = []
        for dataset in encoded_datasets:
            n_batch = len(dataset)
            input_ids = np.zeros((n_batch, num_choices, max_seq_length), dtype=np.int64)
            mc_token_ids = np.zeros((n_batch, num_choices), dtype=np.int64)
            lm_labels = np.full((n_batch, num_choices, max_seq_length), fill_value=-1, dtype=np.int64)
            mc_labels = np.zeros((n_batch,), dtype=np.int64)
            for i, data, in enumerate(dataset):
                q, mc_label = data[0], data[-1]
                choices = data[1:-1]
                for j in range(len(choices)):
                    _truncate_seq_pair(q, choices[j], max_seq_length - 3)
                    qa = [start_token] + q + [delimiter_token] + choices[j] + [clf_token]
                    input_ids[i, j, :len(qa)] = qa
                    mc_token_ids[i, j] = len(qa) - 1
                    lm_labels[i, j, :len(qa) - 1] = qa[1:]
                mc_labels[i] = mc_label
            all_inputs = (input_ids, mc_token_ids, lm_labels, mc_labels)
            tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
        return tensor_datasets

    def tokenize_and_encode(tokenizer, obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, int):
            return obj
        else:
            return list(tokenize_and_encode(tokenizer, o) for o in obj)

    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    tokenizer.add_tokens(GPT_SPECIAL_TOKENS)
    special_tokens_ids = tokenizer.convert_tokens_to_ids(GPT_SPECIAL_TOKENS)

    dataset = load_qa_dataset(statement_jsonl_path)
    examples_ids = [data[0] for data in dataset]
    dataset = [data[1:] for data in dataset]  # discard example ids
    num_choices = len(dataset[0]) - 2

    encoded_dataset = tokenize_and_encode(tokenizer, dataset)

    (input_ids, mc_token_ids, lm_labels, mc_labels), = pre_process_datasets([encoded_dataset], num_choices, max_seq_length, *special_tokens_ids)
    return examples_ids, mc_labels, input_ids, mc_token_ids, lm_labels




######################### BERT/XLNet/Roberta loader utils #########################
class InputExample(object):

    def __init__(self, example_id, question, contexts, endings, label=None):
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label

class InputFeatures(object):

    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'output_mask': output_mask,
            }
            for input_ids, input_mask, segment_ids, output_mask in choices_features
        ]
        self.label = label

def read_examples(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        examples = []
        for line in tqdm(f.readlines()):
            json_dic = json.loads(line)
            label = ord(json_dic["answerKey"]) - ord("A") if 'answerKey' in json_dic else -100
            contexts = json_dic["question"]["stem"]
            if "para" in json_dic:
                contexts = json_dic["para"] + " " + contexts
            if "fact1" in json_dic:
                contexts = json_dic["fact1"] + " " + contexts
            examples.append(
                InputExample(
                    example_id=json_dic["id"],
                    contexts=[contexts] * len(json_dic["question"]["choices"]),
                    question="",
                    endings=[ending["text"] for ending in json_dic["question"]["choices"]],
                    label=label
                ))
    return examples

def simple_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, debug=False):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    concepts_by_sents_list = []
    for ex_index, example in tqdm(enumerate(examples), total=len(examples), desc="Converting examples to features"):
        if debug and ex_index >= debug_sample_size:
            break
        choices_features = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            ans = example.question + " " + ending

            encoded_input = tokenizer(context, ans, padding="max_length", truncation=True, max_length=max_seq_length, return_token_type_ids=True, return_special_tokens_mask=True)
            input_ids = encoded_input["input_ids"]
            output_mask = encoded_input["special_tokens_mask"]
            input_mask = encoded_input["attention_mask"]
            segment_ids = encoded_input["token_type_ids"]

            assert len(input_ids) == max_seq_length
            assert len(output_mask) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((input_ids, input_mask, segment_ids, output_mask))
        label = label_map.get(example.label, -100)
        features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

    return features, concepts_by_sents_list

def load_bert_xlnet_roberta_input_tensors(statement_jsonl_path, max_seq_length, debug, tokenizer, debug_sample_size):

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.bool)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label

    examples = read_examples(statement_jsonl_path)
    num_workers = 60
    if num_workers <= 1:
        features, concepts_by_sents_list = simple_convert_examples_to_features(examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer, debug)
    else:
        from copy import deepcopy
        from multiprocessing import Pool
        c_size = len(examples)//num_workers +1
        examples_list   = [examples[i*c_size: (i+1)*c_size]      for i in range(num_workers)]
        label_list_list = [list(range(len(examples[0].endings))) for i in range(num_workers)]
        max_seq_length_list = [max_seq_length      for i in range(num_workers)]
        tokenizer_list      = [deepcopy(tokenizer) for i in range(num_workers)]
        with Pool(num_workers) as p:
            ress = list(p.starmap(simple_convert_examples_to_features, zip(examples_list, label_list_list, max_seq_length_list, tokenizer_list)))
        features, concepts_by_sents_list = [], []
        for res in ress:
            features += res[0]
            concepts_by_sents_list += res[1]

    example_ids = [f.example_id for f in features]
    *data_tensors, all_label = convert_features_to_tensors(features)
    return example_ids, all_label, data_tensors, concepts_by_sents_list
