import argparse
import io
import json
import os
import pickle
import time
import types

import torch


def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def bool_str_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return v


def int_flag(v):
    return int(float(v))


def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def check_file(file):
    return os.path.isfile(file)


def export_config(config, path):
    param_dict = vars(config)
    check_path(path)
    with open(path, 'w') as fout:
        json.dump(param_dict, fout, indent=4)


def import_config(imported_args, existing_args):
    existing_param_dict = vars(existing_args)
    existing_param_dict.update(vars(imported_args))
    config = types.SimpleNamespace(**existing_param_dict)
    return config


def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_net(module):
    for p in module.parameters():
        p.requires_grad = True


def test_data_loader_ms_per_batch(data_loader, max_steps=10000):
    start = time.time()
    n_batch = sum(1 for batch, _ in zip(data_loader, range(max_steps)))
    return (time.time() - start) * 1000 / n_batch


def print_cuda_info():
    print('torch version:', torch.__version__)
    print('torch cuda version:', torch.version.cuda)
    print('cuda is available:', torch.cuda.is_available())
    print('cuda device count:', torch.cuda.device_count())
    print("cudnn version:", torch.backends.cudnn.version())


def move_tensor(t, device):
    if type(t) == torch.Tensor:
        return t.to(device)
    elif type(t) == list:
        return [move_tensor(x, device) for x in t]
    else:
        return t


def freeze_params(params):
    for p in params:
        p.requires_grad = False


def unfreeze_params(params):
    for p in params:
        p.requires_grad = True


def save_pickle(data, data_path):
    check_path(data_path)
    with open(data_path, "wb") as f:
        pickle.dump(data, f, protocol=4)


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_json(data, file_path):
    check_path(file_path)
    with open(file_path, "w") as f:
        json.dump(data, f, default=set_default)


def save_json_pretty(data, file_path):
    """save formatted json, use this one for some json config files"""
    check_path(file_path)
    with open(file_path, "w") as f:
        f.write(json.dumps(data, indent=4, sort_keys=True, default=set_default))


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def append_filename(filename, appendix):
    name, ext = os.path.splitext(filename)
    return "{name}_{uid}{ext}".format(name=name, uid=appendix, ext=ext)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: 
            return super().find_class(module, name)


def map_wrapper(*args, **kwargs):
    return map(*args)


def sort_dict(d):
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}


def sort_and_normalize_dict(d):
    s = sum(d.values())
    return {k: v / s for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}