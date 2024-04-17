import json
import torch
import numpy as np
import numbers

from utils.utils import get_tensor_mean_min_max_str
from utils.log import log


def str_ellipsis(s, max_len=80):
    partition = 0.3
    front_part = int(max_len * partition)
    end_part = int(max_len * (1 - partition))
    return s if len(s) <= max_len else s[:front_part] + "..." + s[-end_part:]


def get_dict_depth(d, depth=0):
    ret = 0
    if isinstance(d, dict):
        ret = depth
        for k in d:
            ret = max(ret, get_dict_depth(d[k], depth+1))
    else:
        return ret
    return ret

def json_dumps(d: dict):
    return json.dumps(d, indent=4).replace(
                    "true", "True").replace("false", "False").replace("null", "None")

def dict_to_string_join(d, sep="\n", mid=", "):
    ret = ""
    key_list = list(d.keys())
    n = len(key_list)
    for i in range(n):
        k = key_list[i]
        ret += "{}: {}{}".format(str(k),
                                   str(d[k]), mid if i != n-1 else sep)
    return ret



def dict_to_string(d, k='', depth=0, max_depth=0, mmm=False, full_name=True):
    displayed = False
    if depth == 0:
        max_depth = get_dict_depth(d)
    ret = ""
    k_str = (str(k) if k != None else '')
    formatter = "{{}}{{:<{}}} {{:<{}}} {{:>{}}}".format(
        64 + (max_depth - depth) * 8, 24, 32)
    if isinstance(d, torch.Tensor):
        shapes = list(d.shape)
        dim = len(shapes)
        k_str += " ({})".format(d.device)
        type_str = f"{str(list(d.shape))} ({str(d.dtype)})"
        if dim == 0 or (dim == 1 and shapes[0] <= 10):
            ret = formatter.format(
                '\t' * int(depth), k_str, type_str, str(d.tolist()))
        elif d.dtype == torch.float64 or d.dtype == torch.float32 or d.dtype == torch.float16 or d.dtype == torch.uint8:
            ret = formatter.format('\t' * int(depth), k_str, type_str, get_tensor_mean_min_max_str(d.float(), "mmm") if mmm else "")
        else:
            ret = formatter.format(
                '\t' * int(depth), k_str, type_str, str(d.dtype))
    elif isinstance(d, np.ndarray):
        shapes = list(d.shape)
        dim = len(shapes)
        if dim == 0:
            ret = formatter.format(
                '\t' * int(depth), k_str, "dim:{}, {}".format(dim, str(list(d.shape))), d)
        else:
            ret = formatter.format(
                '\t' * int(depth), k_str, "dim:{}, {}".format(dim, str(list(d.shape))), "")
    elif isinstance(d, list):
        # if (isinstance(d[0], numbers.Number) or isinstance(d[0], str)):
        if len(d) == 0:
            ret = formatter.format('\t' * depth, k_str, str(type(d)), str(d))
            displayed = True
        elif len(d) < 5:
            if (type(d[0]) == numbers.Number or type(d[0]) == str):
                ret = formatter.format('\t' * depth, k_str, str(type(d)), str(d))
                displayed = True
            elif not isinstance(d[0], list) and not isinstance(d[0], dict):
                ret = formatter.format(
                    '\t' * depth, k_str, str(type(d))+f"({len(d)})", "item type: {}".format(str(type(d[0]))))
                # displayed = True
                
    elif isinstance(d, int) or isinstance(d, float):
        ret = formatter.format('\t' * depth, k_str, str(type(d)), d)
    elif isinstance(d, str):
        ret = formatter.format('\t' * depth, k_str,
                               str(type(d)), "\"{}\"".format(d))
    else:
        ret = formatter.format('\t' * depth, k_str, str(type(d)), "")

    
    if isinstance(d, dict):
        if not full_name:
            ret += '\t' * int(depth) + k_str
        for name in d.keys():
            if full_name:
                k_str = f"{k}.{str(name)}"
            else:
                k_str = name
            ret += "\n" + \
                dict_to_string(d[name], k_str, depth + 1, max_depth, mmm=mmm, full_name=full_name)
    elif isinstance(d, list) and not displayed:
        if not full_name:
            ret += '\t' * int(depth) + k_str
        for i in range(len(d)):
            if full_name:
                k_str = f"{k}[{str(i)}]"
            else:
                k_str = f"[{str(i)}]"
            ret += "\n" + \
                dict_to_string(d[i], k_str, depth + 1, max_depth, mmm=mmm, full_name=full_name)
    return ret
