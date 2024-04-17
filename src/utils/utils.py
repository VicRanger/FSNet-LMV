import os
import re
import shutil
import matplotlib.pyplot as plt
import torch
import numpy as np
from utils.log import log


def add_metaname(ins, base):
    base_class = base
    if getattr(ins, "full_name", None) is not None:
        ins.full_name = '{}_{}{}'.format(ins.full_name, base_class.class_name, base_class.cnt_instance)
    else:
        setattr(ins,"full_name",'{}{}'.format(base_class.class_name, base_class.cnt_instance))
        setattr(ins,"name",'{}{}'.format(base_class.class_name, base_class.cnt_instance))
    base_class.cnt_instance += 1


def del_dict_item(data: dict, k: str) -> dict:
    del data[k]
    return data


def del_data(data):
    if isinstance(data, dict):
        key_list = list(data.keys())
        for k in key_list:
            data = del_dict_item(data, k)
        return data
    elif isinstance(data, list):
        for i in range(len(data)):
            # FIXME: may not work
            data[i] = del_data(data[i])
        return data
    elif isinstance(data, torch.Tensor):
        del data
    else:
        del data


class Accumulator:

    def __init__(self):
        self.data = None
        self.last_data = None
        self.cnt = 0

    def add(self, data):
        if torch.isinf(torch.tensor(data)) or torch.isnan(torch.tensor(data)):
            return
        if self.cnt == 0:
            self.data = data
        else:
            self.data += data
        self.last_data = data
        self.cnt += 1

    def get(self):
        if self.cnt == 0:
            return "no data."
        return self.data / self.cnt

    def reset(self):
        self.data = None
        self.cnt = 0


def time_format(seconds: int):
    if seconds is not None:
        seconds = int(seconds)
        d = seconds // (3600 * 24)
        h = seconds // 3600 % 24
        m = seconds % 3600 // 60
        s = seconds % 3600 % 60
        if d > 0:
            return '{:02d}:{:02d}:{:02d}:{:02d}'.format(d, h, m, s)
        elif h > 0:
            return '{:02d}:{:02d}:{:02d}'.format(h, m, s)
        elif m > 0:
            return '{:02d}:{:02d}'.format(m, s)
        elif s > 0:
            return '{:02d}s'.format(s)
    return '-'


def create_dir(path):
    if not (os.path.exists(path)):
        os.makedirs(path)
        return True
    return False


def write_text_to_file(path, output, mode="w"):
    f = open(path, mode)
    f.write(output)
    f.close()


def add_at_dict_front(d, key, value):
    new_d = {}
    new_d[key] = value
    for k in d.keys():
        new_d[k] = d[k]
    return new_d


def remove_all_in_dir(path):
    if not (os.path.exists(path)):
        return
    content = os.listdir(path)
    if len(content) > 0:
        for item in content:
            total_path = path + "/" + item
            if os.path.isdir(total_path):
                shutil.rmtree(total_path, ignore_errors=True)
            else:
                os.remove(total_path)
        return True
    return False


def get_file_component(file_path):
    file_path = file_path.replace("\\", "/")
    result = re.match("(.*)/(.*)[.](.*)", file_path).groups()
    if len(result) != 3:
        raise RuntimeError(
            "cant get correct component of {}, result:{}".format(file_path, result))
    return {
        'path': result[0],
        'filename': result[1],
        'suffix': result[2],
    }


def arr_is_in_arr(a, b):
    '''check if items of arr A is all in arr B

    Args:
        a: arr A
        b: arr B
    Returns:
        (1): bool, if items of arr A is all in arr B
        (2): <type of arr>, first invalid item
    '''
    for a_item in a:
        if a_item not in b:
            return False, a_item
    return True, None


def deal_with_module(module, act="relu"):
    torch.nn.init.kaiming_uniform_(
        module.weight, nonlinearity=act)
    module.bias.data.fill_(0)


def get_tensor_mean_min_max(t):
    # fp16 will always output nan when using nanmean
    if t.numel() == 0:
        return 0, 0, 0
    if t.dtype == torch.float16:
        return t.float().mean().half(), t.min(), t.max()
    else:
        return t.mean(), t.min(), t.max()


def get_tensor_mean_min_max_str(t, name="", mode="f"):
    return "{{}}: {{:.3{}}} {{:.3{}}} {{:.3{}}}".format(mode, mode, mode).format(name, *get_tensor_mean_min_max(t))


def show(img):
    plt.imshow(img)
    plt.show()


def to_img(arr):
    data = arr.permute(1, 2, 0)
    img = (data.detach().cpu().numpy() * 255.0).astype(np.uint8)
    return img
