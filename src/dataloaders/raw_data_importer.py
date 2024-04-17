import copy
from glob import glob
import json
import math
import os
from select import select
import threading
import time
import re
import multiprocessing as mp
from tqdm import tqdm
import torch
from utils.utils import arr_is_in_arr, create_dir, del_dict_item, get_file_component, write_text_to_file
from .dataset_base import dispatch_task_by_metadata
from utils.dataset_utils import compose_scene_color, create_brdf_color, create_cross_sample, create_de_color, \
    create_history_frame, create_history_warped_buffer, create_history_warped_scene_color_cross, create_occlusion_mask, create_scene_color, \
    create_scene_color_no_st, create_shadow_discontinuity_mask, create_sky_color, create_skybox_mask, create_st_color, \
    create_warped_buffer, transform_direction, transform_direction_image, transform_position_image, write_npz, write_torch

from utils.log import log
from utils.str_utils import dict_to_string
from utils.buffer_utils import buffer_data_to_vis, buffer_raw_to_data, fix_dmdl_color_zero_value, read_buffer, write_buffer
from utils.warp import get_merged_motion_vector_from_last, warp
from .dataset_base import MetaData
from utils.parser_utils import parse_buffer_name, parse_find_dict, parse_flat_dict


def get_augmented_buffer(augmented_output, buffer_config, data, last_data=[], allow_skip=False, with_batch=False, history_data_check=True) -> None:
    if augmented_output is None:
        return
    demodulation_mode = buffer_config['demodulation_mode']
    augmented_data_recipe = buffer_config['augmented_data_recipe']
    
    for key_augmented in augmented_output:
        ret = parse_buffer_name(key_augmented)
        pref = ret['sample']
        buffer_name = ret['buffer_name']
        his_id = ret['his_id']
        postf = ret['postf']
        p_buffer_name = pref + buffer_name
        p_buffer_name_p = pref + buffer_name + postf
        if key_augmented in data.keys():
            continue


        if p_buffer_name not in augmented_data_recipe.keys():
            raise Exception(f'{p_buffer_name} can\'t be found in augmented_data_recipe ({list(augmented_data_recipe.keys())})')
        num_history = augmented_data_recipe[pref + buffer_name].get('num_history', 0)

        if p_buffer_name not in augmented_data_recipe.keys():
            raise Exception("creating {}, found \"{}\" is not in recipe.keys:{}".format(
                key_augmented, p_buffer_name, list(augmented_data_recipe.keys())))
        flag, invalid_item = arr_is_in_arr(
            augmented_data_recipe[p_buffer_name]['dep'], data.keys())

        err_msg = ""
        if not flag:
            err_msg = "creating {}, found {} (all_dep:{}) isn\'t in data.keys: {}".format(
                key_augmented, invalid_item, augmented_data_recipe[p_buffer_name]['dep'], data.keys())

        if history_data_check:
            if (his_id is not None and his_id > num_history) or (his_id is None and flag and len(last_data) < num_history):
                flag = False
                err_msg = "creating {}, the length of last_data is {}, but requirement is {}".format(
                    key_augmented, len(last_data), num_history)

        for i in range(num_history):
            if not history_data_check:
                break
            if flag:
                if buffer_name.startswith("history_"):
                    key_augmented = f'{pref}{buffer_name}_{his_id}'
                    if (his_id is not None and i == his_id) or his_id is None:
                        flag, invalid_item = arr_is_in_arr(
                            augmented_data_recipe[p_buffer_name]['dep_history'][i], last_data[i].keys())
                else:
                    flag, invalid_item = arr_is_in_arr(
                        augmented_data_recipe[p_buffer_name]['dep_history'][i], last_data[i].keys())
                if not flag:
                    err_msg = "creating \"{}\" as \"{}\", found {} isn\'t in last_data[{}].keys: {}".format(
                        key_augmented, p_buffer_name, invalid_item, i, last_data[i].keys())
            else:
                break

        if not flag:
            if not allow_skip:
                raise Exception(
                    f"\n{err_msg}\naugmented_output:\n{augmented_output}\ndata_dict:" +
                    f"\n{dict_to_string(data)}\nrecipe:\n{dict_to_string(augmented_data_recipe[p_buffer_name])}")
            else:
                continue

        augmented_data = None

        if buffer_name == "black3":
            augmented_data = torch.zeros_like(data['base_color' + postf])
        elif buffer_name == "white1":
            augmented_data = torch.ones_like(data['base_color' + postf][:1])
        elif buffer_name == "stencil":
            augmented_data = torch.zeros_like(data['base_color' + postf][:1])

        elif buffer_name.startswith("history_warped_"):
            name = buffer_name.replace("history_warped_", "")
            if his_id is not None:
                data[f"{pref}{buffer_name}_{his_id}{postf}"] = create_warped_buffer(
                    last_data[his_id][pref + name + postf], data[f'{pref}merged_motion_vector_{his_id}{postf}'],
                    mode="bilinear", padding_mode="border")
            else:
                for i in range(0, num_history, 1):
                    for i in buffer_config['index']:
                        if i < len(last_data) - 1:
                            continue
                        data[f"{pref}{buffer_name}_{i}{postf}"] = create_warped_buffer(
                            last_data[i][pref + name + postf], data[f'{pref}merged_motion_vector_{i}{postf}'],
                            mode="bilinear", padding_mode="border")

        elif buffer_name.startswith("history_"):
            history_name = buffer_name
            buffer_name = buffer_name.replace("history_", "")
            if his_id is not None:
                data[f"{pref}{history_name}_{his_id}{postf}"] = create_history_frame(
                    last_data, f'{pref}{buffer_name}{postf}', index=his_id)
            else:
                for i in buffer_config['index']:
                    if i > len(last_data) - 1:
                        continue
                    data[f"{pref}{history_name}_{i}{postf}"] = create_history_frame(
                        last_data, f'{pref}{buffer_name}{[postf]}', index=i)
        elif buffer_name == "merged_motion_vector":
            mv = data[f'{pref}merged_motion_vector_0{postf}'] = data[f'{pref}motion_vector{postf}']
            num_history = len(last_data)
            for history_ind in range(1, num_history):
                mv = get_merged_motion_vector_from_last(
                        last_data[history_ind-1][f'{pref}motion_vector{postf}'], mv, residual=mv)
                if with_batch:
                    data[f'{pref}merged_motion_vector_{history_ind}{postf}'] = mv
                else:
                    data[f'{pref}merged_motion_vector_{history_ind}{postf}'] = mv[0]


        elif buffer_name.startswith("history_multi_warped_"):
            name = buffer_name.replace("history_multi_warped_", "")
            for i in range(num_history):
                data[f'{pref}{buffer_name}_{i}{postf}'] = create_history_warped_buffer(
                    data, last_data, i, name, prefix=pref, postfix=postf,
                    mode="bilinear", padding_mode="border")

        elif buffer_name == "scene_color":
            augmented_data = create_scene_color(data[pref + 'scene_color_no_st' + postf],
                                                data[pref + 'st_color' + postf], data[pref + 'st_alpha' + postf])

        elif buffer_name == "scene_color_no_st":
            augmented_data = create_scene_color_no_st(data[pref + 'scene_color' + postf],
                                                      data[pref + 'st_color' + postf], data[pref + 'st_alpha' + postf])

        elif buffer_name == "st_color":
            augmented_data = create_st_color(data[pref + 'scene_color' + postf],
                                             data[pref + 'scene_color_no_st' + postf], data[pref + 'st_alpha' + postf])

        elif buffer_name == "st_alpha":
            augmented_data = torch.ones_like(data[pref + 'depth' + postf])

        elif buffer_name == 'brdf_color':
            augmented_data = create_brdf_color(data[pref + 'roughness' + postf], data[pref + 'nov' + postf],
                                               data[pref + 'base_color' + postf], data[pref +
                                                                                       'metallic' + postf], data[pref + 'specular' + postf],
                                               skybox_mask=data[pref + 'skybox_mask' + postf])
        elif buffer_name == 'dmdl_color':
            if demodulation_mode == 'base':
                augmented_data = fix_dmdl_color_zero_value(data[pref + 'base_color'])
            elif demodulation_mode == 'brdf':
                key_brdf_color = key_augmented.replace("dmdl_color", "brdf_color")
                if key_brdf_color in data.keys():
                    augmented_data = data[key_brdf_color]
                else:
                    augmented_data = create_brdf_color(data[pref + 'roughness' + postf],
                                                       data[pref + 'nov' + postf],
                                                       data[pref + 'base_color' + postf],
                                                       data[pref + 'metallic' + postf],
                                                       data[pref + 'specular' + postf],
                                                       skybox_mask=data[pref + 'skybox_mask' + postf])
            else:
                raise Exception(
                    f'dmdl_color only supports "base", "brdf", but buffer_config.demodulation_mode="{demodulation_mode}"')


        elif buffer_name == 'scene_light_no_st':
            sc = data[pref + buffer_name.replace('scene_light', 'scene_color') + postf]
            augmented_data = create_de_color(sc, data[pref + 'dmdl_color' + postf],
                                             skybox_mask=data[pref + 'skybox_mask' + postf], sky_color=data[pref + 'sky_color' + postf], fix=True)
        elif buffer_name == 'scene_light':
            sc = data[pref + buffer_name.replace('scene_light', 'scene_color') + postf]
            augmented_data = create_de_color(sc, data[pref + 'dmdl_color' + postf],
                                             skybox_mask=data[pref + 'skybox_mask' + postf], sky_color=data[pref + 'sky_color' + postf], fix=True)

        elif buffer_name == 'skybox_mask':
            sky_depth = data.get(pref + 'sky_depth' + postf, None)
            augmented_data = create_skybox_mask(
                data[pref + 'depth' + postf], data[pref + 'base_color' + postf], sky_depth=sky_depth)

        elif buffer_name == "sky_color":
            augmented_data = create_sky_color(
                data[pref + 'scene_color_no_st' + postf], data[pref + 'skybox_mask' + postf])

        else:
            raise NotImplementedError(
                "{} is not supported for augmented_buffer".format(key_augmented))

        if augmented_data is not None:
            data[key_augmented] = augmented_data


def compress_buffer(data, data_type='fp16'):
    '''
    `example input`
    '''
    raw_data = copy.deepcopy(data)
    for k in raw_data.keys():
        if not (isinstance(raw_data[k], torch.Tensor)):
            continue
        raw_data[k] = tensor_as_type_str(raw_data[k], type_str=data_type)
    return raw_data


def get_extend_buffer(data: dict, part_name: str, buffer_config: dict, last_datas=[], start_cutoff=5) -> dict:
    '''
    ### input:
    `data`: input buffer dict

    `part_name`: part to extend

    `buffer_config`: config for extend rules

    `last_data`: history input buffer dict

    `start_cutoff`: not perform because of lack of history data

    ### output:
    `0: extended data, like {'<buffer_name1>': torch.Tensor, '<buffer_name2>': torch.Tensor}
    '''
    metadata = MetaData(data['metadata']["scene_name"], data['metadata']['index'])
    ret = {}
    if metadata.index < start_cutoff:
        log.debug("metadata: {} doesnt have previous data, skip.".format(
            metadata.__str__()))
        return ret

    part = buffer_config['part'][part_name]

    get_augmented_buffer(part.get('augmented_data', []) + part['buffer_name'],
                         buffer_config,
                         data, last_data=last_datas,
                         allow_skip=False)

    ret = {}
    for buffer_name in part['buffer_name']:
        if buffer_name not in data.keys():
            raise KeyError(f'{buffer_name} not in data.keys(), keys:{list(data.keys())}')
        if isinstance(data[buffer_name], torch.Tensor):
            ret[buffer_name] = tensor_as_type_str(
                data[buffer_name], part['type'])
        else:
            ret[buffer_name] = data[buffer_name]
        if isinstance(ret[buffer_name], torch.Tensor) and\
                (torch.isinf(ret[buffer_name]).any() or torch.isnan(ret[buffer_name]).any()):
            log.debug(dict_to_string(data, mmm=True))
            log.debug(dict_to_string(ret, mmm=True))
            log.warn(f"{'!'*10} an nan or inf occur in motion vector in {metadata.__str__()} {'!'*10}")
    return ret


def tensor_as_type_str(tensor, type_str):
    def to_fp16(data):
        data = data.type(torch.float16).cuda()
        data = torch.clamp(data, -65504, 65504).cpu()
        return data
    ops = {
        'fp16': lambda x: to_fp16(x),
        'fp32': lambda x: x.type(torch.float32)
    }
    return ops[type_str](tensor)


def dualize_engine_buffer(engine_buffer, post_fixes, exclusion_names=[]):
    new_engine_buffer = {}
    for k in engine_buffer.keys():
        if k in exclusion_names:
            continue
        for pf in post_fixes:
            new_engine_buffer[k + pf] = engine_buffer[k]
    return new_engine_buffer


def dualize_output_buffer(output_buffer, post_fixes, exclusion_names=[]):
    new_output_buffer = {}
    for k in output_buffer.keys():
        if k in exclusion_names:
            continue
        for pf in post_fixes:
            new_output_buffer[k + pf] = {
                "origin": output_buffer[k]["origin"] + pf.upper(),
                "channel": output_buffer[k]["channel"]
            }
    return new_output_buffer


def dualize_buffer_list(buffer_list, post_fixes, exclusion_names=[]):
    new_list = []
    for item in buffer_list:
        if item in exclusion_names:
            continue
        for pf in post_fixes:
            new_list.append(item + pf)
    return new_list


def dualize_augmented_data_recipe(augmented_data_recipe, post_fixes, exclusion_names=[]):
    new_recipe = {}
    for k in augmented_data_recipe.keys():
        if k in exclusion_names:
            continue
        for pf in post_fixes:
            new_recipe[k + pf] = copy.deepcopy(augmented_data_recipe[k])
            new_recipe[k + pf]["dep"] = dualize_buffer_list(
                augmented_data_recipe[k]["dep"], post_fixes, exclusion_names=exclusion_names)
            num_history = augmented_data_recipe[k].get("num_history", 0)
            assert (num_history == len(
                augmented_data_recipe[k].get('dep_history', [])))
            for i in range(num_history):
                new_recipe[k + pf]["dep_history"][i] = dualize_buffer_list(
                    augmented_data_recipe[k]["dep_history"][i], post_fixes, exclusion_names=exclusion_names
                )
    return new_recipe


def task_wrapper(ins, scene, start_index, end_index, file_index, idx):
    log.debug("start wrapper {} {} {} {} {} {}".format(
        ins, scene, start_index, end_index, file_index, idx))
    ins.export_patch_range(scene, start_index, end_index, file_index, idx)


def dualize_buffer_config(buffer_config):
    exclusion_names = buffer_config['dual_exclusion']
    buffer_config['engine_buffer'] = dualize_engine_buffer(
        buffer_config['engine_buffer'], ["_L", "_R"], exclusion_names=exclusion_names)
    buffer_config['output_buffer'] = dualize_output_buffer(
        buffer_config['output_buffer'], ["_l", "_r"], exclusion_names=exclusion_names)
    buffer_config['history_buffer'] = dualize_output_buffer(
        buffer_config['history_buffer'], ["_l", "_r"], exclusion_names=exclusion_names)
    buffer_config['augmented_data'] = dualize_buffer_list(
        buffer_config['augmented_data'], ["_l", "_r"], exclusion_names=exclusion_names)
    buffer_config['augmented_data_on_the_fly'] = dualize_buffer_list(
        buffer_config['augmented_data_on_the_fly'], ["_l", "_r"], exclusion_names=exclusion_names)
    buffer_config['augmented_data_recipe'] = dualize_augmented_data_recipe(
        buffer_config['augmented_data_recipe'], ["_l", "_r"], exclusion_names=exclusion_names)
    for part in buffer_config['addition_part']:
        part['augmented_data'] = dualize_buffer_list(
            part['augmented_data'], ["_l", "_r"], exclusion_names=exclusion_names)
        part['buffer_name'] = dualize_buffer_list(
            part['buffer_name'], ["_l", "_r"], exclusion_names=exclusion_names)


class UE4RawDataLoader:
    job_config = {}
    buffer_config = {}
    flow_estimator = None

    def __init__(self, in_buffer_config, in_job_config):
        log.info("start UE4RawDataLoader init")
        if in_buffer_config['dual']:
            dualize_buffer_config(in_buffer_config)
        self.job_config = in_job_config
        self.buffer_config = in_buffer_config
        self.data = dict()
        self.metadata = dict()

        for s in self.job_config['scene']:
            path = self.job_config['import_path'] + s
            num = self.check_files(path)
            self.data[s] = []
            self.metadata[s] = []
            for i in tqdm(range(num), ncols=64):
                self.metadata[s].append(MetaData(s, i))
            log.info("{} processed. total frame: {}.".format(
                s, len(self.metadata[s])))

    def check_files(self, path, unsafe=False):
        num = -1
        for item in self.buffer_config['engine_buffer'].keys():
            file_name_list = glob(
                "{}/{}/*[0-9].*".format(path, str(item)))
            cur_num = len(file_name_list)
            if num != -1 and cur_num != num:
                log.error("found error when exporting the {}".format(path))
                if not unsafe:
                    raise FileNotFoundError(
                        "{} length({}) is not same as required({}). ".format(item, cur_num, num))
            num = cur_num
        log.info("path:{}, pre-check passed, num:{}.".format(path, num))
        return num

    def parse_buffer(self, buffers, directory, ind):
        tmp_data = {}
        for buffer_name in buffers.keys():
            origin = buffers[buffer_name]['origin']
            if origin not in self.buffer_config['engine_buffer'].keys():
                raise KeyError(
                    "{} is not in engine_buffer. keys:{}".format(origin, self.buffer_config['engine_buffer'].keys()))
            suffix = self.buffer_config['engine_buffer'][origin]['suffix']
            channel = buffers[buffer_name]['channel']
            path = self.job_config['pattern'].format(
                directory, origin, ind, suffix)
            buffer_data = read_buffer(path=path, channel=channel)

            if torch.isinf(buffer_data).any() or torch.isnan(buffer_data).any():
                log.warn(f'{"="*10 + "warning" + "="*10}\n there is inf or nan in "{path}"')

            buffer_data[torch.isinf(buffer_data)] = 0.0
            buffer_data[torch.isnan(buffer_data)] = 0.0

            tmp_data[buffer_name] = buffer_raw_to_data(
                buffer_data, buffer_name)

        return tmp_data

    def parse_scene(self, directory, ind):
        f = open(self.job_config['pattern'].format(
            directory, self.job_config['scene_info_name'], ind, "txt"))
        lines = [x.strip() for x in f.readlines()]
        # log.debug(lines)
        i = 0
        dict_name, i = parse_find_dict(lines, i)
        ret, i = parse_flat_dict(lines, "", i + 1)
        return ret

    def get_patch(self, scene, index):
        import_path = self.job_config['import_path'] + scene
        tmp_data = []
        tmp_data = self.parse_buffer(
            self.buffer_config['output_buffer'], import_path, self.metadata[scene][index].index)
        tmp_data['metadata'] = self.metadata[scene][index].to_dict()
        return tmp_data

    def export_patch(self, metadata: MetaData):
        suffix = ""
        scene = metadata.scene_name
        index = metadata.index
        file_path_template = "{}/{}{}/{{}}/{{}}.{{}}".format(
            self.job_config['export_path'], scene, suffix)
        overwrite = self.job_config.get('overwrite', False)

        metadata_part = self.buffer_config['metadata_part']

        ''' overwrite handler '''
        if not overwrite and os.path.exists(file_path_template.format(metadata_part, index, "npz")):
            log.debug("{} exists.".format(
                file_path_template.format(metadata_part, index, "npz")))
            return

        origin_data = self.get_patch(scene, index)
        ret = {}

        ''' extend process '''
        for part_name in self.buffer_config['basic_part_enable_list']:
            ret[part_name] = get_extend_buffer(origin_data, part_name, self.buffer_config, start_cutoff=0)

        ''' write buffer to disk'''
        export_path = ""
        for part_name in self.buffer_config['basic_part_enable_list']:
            if index == 0:
                log.info(dict_to_string(ret[part_name]))

            export_path = file_path_template.format(part_name, index, "npz")
            path_comp = get_file_component(export_path)
            create_dir(path_comp['path'])
            scene_path = os.path.join(path_comp['path'], "..")
            if not os.path.exists(scene_path + "/" + "{}.txt".format(part_name)):
                write_text_to_file(scene_path + "/" + "{}.txt".format(part_name),
                                   json.dumps(list(ret[part_name].keys()), indent=4).replace(
                                       "true", "True").replace("false", "False").replace("null", "None"), "w")
            ret[part_name] = compress_buffer(ret[part_name], self.buffer_config['part'][part_name]['type'])
            log.debug(dict_to_string(ret[part_name], mmm=True))
            write_npz(export_path, ret[part_name])
        log.debug("patch \"{}\" exported. patch:{}".format(MetaData(scene, index), export_path))

    def export(self):
        for s in self.job_config['scene']:
            test_config = self.job_config.get('test_config', None)
            if test_config is not None and test_config['enable']:
                num = test_config['nums']
                tmp_metadatas = self.metadata[s][:num]
            else:
                tmp_metadatas = self.metadata[s]
            dispatch_task_by_metadata(self.export_patch, tmp_metadatas,
                                      num_thread=self.job_config.get('num_thread', 0))

    def test_task(self, metadata):
        print(f"start range: {metadata}")
