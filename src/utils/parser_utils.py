import copy
import json
import argparse
import os
import re
from utils.str_utils import dict_to_string
from utils.log import log
import numpy as np
import torch


def parse_buffer_name(key_augmented) -> dict:
    '''
    ### return:
    `pref`, `buffer_name`, `his_id`, `postf`
    '''

    # r"(.+)" + \
    # r"((?:_[a-z]+)(_[a-z]+[a-z\d]*))" + \
    template = r"^(?P<sample>[d s a u][\d]+_|aa_){0,1}" + \
        r"(?P<his_pref>history_)*" + \
        r"(?P<his_warp>warped_)*" + \
        r"(?P<basic_element>(?:[a-z]+[a-z\d]*)(?:_{1,2}[a-z]+)*(?:_[a-z]+[\d]+)*)" + \
        r"_{0,1}(?P<his_id>[\d]+){0,1}" + \
        r"(?P<method>_extranet){0,1}" +\
        r"(?P<postf>[_lr]){0,1}$"
    # log.debug(template)
    # log.debug(key_augmented)
    key_list = ['sample', 'his_pref', 'his_warp', 'basic_element', 'his_id', 'method', 'postf']
    res = re.search(
        template, key_augmented)
    if res:
        ret = {k: (str(res.group(k)) if str(res.group(k)) != "None" else "") for k in key_list}
        # log.debug(dict_to_string(ret))
        ''' method (eg: _extranet)'''
        # method = res.group(4)
        # postf = res.group(5)
        ret['his_id'] = int(ret['his_id']) if ret['his_id'] else None # type: ignore
        ret['buffer_name'] = ret['his_pref'] + ret['his_warp'] + ret['basic_element'] + ret['method']
    else:
        raise NameError(
            "{} is not passed for get_augmented_buffer pre-check.".format(key_augmented))
    # log.debug(f'"{pref}" "{buffer_name}" "{his_id}" "{postf}"')
    return ret


def create_json_parser(file_name):
    if not (os.path.exists(file_name)):
        return {}
    baseConfigFile = open(file_name, "r")
    content = baseConfigFile.readlines()
    content = "".join([s.strip() for s in content])
    configDict = json.loads(content)
    baseConfigFile.close()
    return configDict


def create_py_parser(file_name):
    if not (os.path.exists(file_name)):
        raise FileNotFoundError("file: '{}' not found.".format(file_name))
    baseConfigFile = open(file_name, "r")
    content = [l.strip() for l in baseConfigFile.readlines()]
    eval_str = "\n".join(content)
    configDict = eval(eval_str)
    baseConfigFile.close()
    return configDict


def create_parser(inBaseConfigDict):
    # generate a parser
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    for k in inBaseConfigDict:
        parser.add_argument(
            '--' + k, default=inBaseConfigDict[k], type=type(inBaseConfigDict[k]))
    return parser


def overload_parser(in_parser, in_config_dict):
    for k in in_config_dict:
        in_parser.add_argument(
            '--' + k, default=in_config_dict[k], type=type(in_config_dict[k]))
    return in_parser


def overwrite_dict(in_dict, in_config_dict):
    for k in in_config_dict.keys():
        print(k)
        if type(in_config_dict[k]) == dict:
            overwrite_dict(in_dict[k], in_config_dict[k])
        else:
            in_dict[k] = in_config_dict[k]
    return in_dict


def parse_n(lines, cur_i, target_type=int):
    ret = target_type(lines[cur_i])
    return ret, cur_i


def parse_v(lines, cur_i, number=3):
    nums = [float(n) for n in lines[cur_i].split()]
    if len(nums) != number:
        raise ValueError(
            "parse v{} received {} numbers".format(number, len(nums)))
    ret = torch.tensor(nums)
    # log.debug("parse_v{}: {} for {}".format(number, lines[cur_i], ret))
    return ret, cur_i


def parse_m(lines, cur_i, row=4, col=4):
    mat = []
    for i in range(row):
        nums, cur_i = parse_v(lines, cur_i, number=col)
        mat.append(nums)
        cur_i += 1
    ret = torch.tensor(np.array([v.numpy() for v in mat]))
    # log.debug("parse_m{}x{}: {} for {}".format(row, col, lines[cur_i - row + 1:cur_i + 1], ret))
    return ret, cur_i - 1


def parse_flat_dict(lines, name, cur_i):
    n = len(lines)
    ret = {}
    i = cur_i
    prefix = name + "__" if len(name) > 0 else ""
    while i < n:
        item = lines[i].split()
        n_item = len(item)
        if n_item <= 0:
            i += 1
            continue
        if item[0] == "end":
            return ret, i
        if len(item) == 2:
            if item[0] in parse_ops.keys():
                # log.debug("procsess {} {}".format(item[0], item[1]))
                if (item[0] != "dict"):
                    ret[prefix + item[1]], i = parse_ops[item[0]](lines, i + 1)
                else:
                    tmp_dict, i = parse_ops['flat_dict'](
                        lines, prefix + item[1], i + 1)
                    # log.debug(tmp_dict)
                    ret.update(tmp_dict)
            else:
                raise KeyError("{} not in ops".format(item[0]))
        else:
            raise SyntaxError("text '{}' cant be solve".format(lines[i]))
        i += 1
    return ret, i - 1


def parse_dict(lines, cur_i):
    n = len(lines)
    ret = {}
    i = cur_i
    while i < n:
        item = lines[i].split()
        n_item = len(item)
        if n_item <= 0:
            i += 1
            continue
        if item[0] == "end":
            return ret, i
        if len(item) == 2:
            if item[0] in parse_ops.keys():
                # log.debug("procsess {} {}".format(item[0], item[1]))
                ret[item[1]], i = parse_ops[item[0]](lines, i + 1)
            else:
                raise KeyError("{} not in ops".format(item[0]))
        else:
            raise SyntaxError("text '{}' cant be solve".format(lines[i]))
        i += 1
    return ret, i - 1


def parse_find_dict(lines, cur_i):
    i = cur_i
    n = len(lines)
    ret = {}
    while i < n and (len(lines[i]) <= 0 or lines[i].split()[0] != 'dict'):
        i += 1
    dict_name = lines[i].split()[1]
    return dict_name, i


parse_ops = {
    'dict': lambda lines, i: parse_dict(lines, i),
    'flat_dict': lambda lines, name, i: parse_flat_dict(lines, name, i),
    'v3': lambda lines, i: parse_v(lines, i, 3),
    'm4': lambda lines, i: parse_m(lines, i, 4, 4),
    'n': lambda lines, i: parse_n(lines, i, target_type=int)
}
