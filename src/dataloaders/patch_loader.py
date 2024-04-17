import copy
from glob import glob
import json
import os
import re
import time
import math
from zipfile import BadZipfile
import torch
import gc
from utils.buffer_utils import write_buffer
# from .patch_cropper import crop
from utils.utils import del_dict_item, write_text_to_file
from utils.utils import create_dir, get_file_component
from .raw_data_importer import compress_buffer, get_augmented_buffer, get_extend_buffer, parse_buffer_name, tensor_as_type_str
from .raw_data_importer import dualize_buffer_list
from .raw_data_importer import dualize_buffer_config
from utils.dataset_utils import data_to_device, write_npz
from .dataset_base import MetaData, create_metadata_by_glob
import numpy as np
from .raw_data_importer import UE4RawDataLoader
from utils.model_utils import min_max_scalar
from tqdm import tqdm
from utils.str_utils import dict_to_string
from utils.log import log
import multiprocessing as mp
from utils.buffer_utils import aces_tonemapper
from utils.dataset_utils import create_warped_buffer

def mix(a,b,t):
    return a*(1-t)+b*t

g_augmented_data_output = None
def history_extend(data, config):
    global g_augmented_data_output
    history_data_list = data['history_data_list']
    buffer_config = config['buffer_config']
    history_config = config['dataset']['history_config']
    require_list = config['dataset']['require_list']
    
    for ind, history_data in enumerate(history_data_list):
        if len(list(history_data.keys())) <= 0:
            continue
        get_augmented_buffer(
            history_config['augmented_data'][ind],
            buffer_config,
            history_data,
            last_data=[],
            allow_skip=history_config['allow_skip'],
            with_batch=True)
    
        
    if len(require_list) > 0 and g_augmented_data_output is None:
        g_augmented_data_output = []
        ''' if data_config isnt empty, we continue generating. '''
        if buffer_config is not None:
            for item in require_list:
                if item not in data.keys():
                    res = parse_buffer_name(item)
                    pref = res['sample']
                    buffer_name = res['buffer_name']
                    his_id = res['his_id']
                    postf = res['postf']
                    ''' temporay exception for history_masked_warped_scene_color_extranet '''
                    if item == "history_masked_warped_scene_color_0_extranet":
                        buffer_name = item
                    elif "extra" in item:
                        continue
                    # log.debug("'{}' '{}' '{}' '{}'".format(item, buffer_name, his_id, pf))
                    if pref + buffer_name + postf in buffer_config['augmented_data_recipe'].keys():
                        g_augmented_data_output.append(item)
                    else:
                        raise NotImplementedError("{}({}) is required but not in augmented_data_recipe.keys ({})".format(
                            pref + buffer_name + postf, item, buffer_config['augmented_data_recipe'].keys()))

    get_augmented_buffer(
        g_augmented_data_output,
        buffer_config,
        data,
        last_data=history_data_list,
        allow_skip=False,
        with_batch=True)
    
    if len(require_list) > 0:
        ret = {}
        for k in data.keys():
            if k in require_list:
                ret[k] = data[k]
        data.clear()
        data = ret
        
    return data


class PatchLoaderBase:
    def __init__(self):
        self.export_path = ""

    def load(self):
        pass


retry_num = 6
sleep_time = 10


class PatchLoader(PatchLoaderBase):
    # use MetaData to load a series of pass of a frame, return tersor data
    cache_dict = {}
    last_cached_key = []

    def __init__(self, data_part, buffer_config={}, cache_num = 8,
                 job_config={}, require_list=[], augmented_data_output=None, with_augment=True):
        super(PatchLoader, self).__init__()
        if buffer_config['dual']:
            dualize_buffer_config(buffer_config)

        self.job_config = job_config
        self.export_path = self.job_config['export_path']
        self.buffer_config = buffer_config
        self.augmented_data_output = augmented_data_output
        self.data_part = data_part
        self.last_data = {}
        self.require_list = require_list
        self._debug_hit = 0 
        self._debug_total = 0
        self.with_augment = with_augment
        self.cache_num = cache_num
        if self.buffer_config['dual']:
            self.require_list = dualize_buffer_list(self.require_list, post_fixes=[
                "_l", "_r"], exclusion_names=self.buffer_config['dual_exclusion'])

        ''' always False'''
        self.gpu = False
        
        self.enable_cache = True
        log.debug(f"require_list: {self.require_list}")

    def load_npz(self, metadata, part_name="metadata"):
        ret = None
        self._debug_total += 1
        if self.enable_cache:
            cache_key = f"{metadata.__str__()}"
            if cache_key in PatchLoader.cache_dict.keys():
                if part_name in PatchLoader.cache_dict[cache_key].keys():
                    ret = copy.deepcopy(PatchLoader.cache_dict[cache_key][part_name])
                    self._debug_hit += 1
        else: cache_key = ""

        if ret is None:
            file_path = self.get_file_path(
                metadata, part_name)
            data = None
            for i in range(retry_num):
                try:
                    data = np.load(file_path, allow_pickle=True)
                    break
                except FileNotFoundError:
                    log.debug("file_path:{}, retry_time: {}".format(file_path, i + 1))
                    time.sleep(sleep_time)
            if data is None:
                log.debug("ERROR. file_path:{}".format(file_path))
                data = np.load(file_path, allow_pickle=True)
            assert data is not None
            ret = []
            for name in data.files:
                if name == 'allow_pickle':
                    continue
                else:
                    item = data[name].item()
                    ret.append(item)
                    break
            ret = ret[0]
            ''' cant work with multi-core dataloader '''

            if self.enable_cache:
                if cache_key not in PatchLoader.cache_dict.keys():
                    PatchLoader.cache_dict[cache_key] = {}
                    PatchLoader.last_cached_key.append(cache_key)
                    if len(PatchLoader.last_cached_key) > self.cache_num:
                        # del_key = list(PatchLoader.cache_dict.keys())[-1]
                        del_key = PatchLoader.last_cached_key[0]
                        # log.debug(f"delete key:{del_key}")
                        del PatchLoader.cache_dict[del_key]
                        del PatchLoader.last_cached_key[0]
                        gc.collect()
                PatchLoader.cache_dict[cache_key][part_name] = copy.deepcopy(ret)

        for k in ret.keys():
            if isinstance(ret[k], torch.Tensor):
                    ret[k] = ret[k].type(torch.float32)

            if 'scene_color_no_st' in k or 'st_color' in k or 'sky_color' in k:
                ret[k] = torch.clamp(ret[k], min=0)
            if 'st_alpha' in k:
                ret[k] = torch.clamp(ret[k], 0.0, 1.0)
        return ret

    def load_data(self, metadata: MetaData, data_part=None):
        data = {}
        if data_part is None:
            data_part = self.data_part
        assert isinstance(data_part, list)
        if len(data_part) <= 0:
            raise ValueError(f"can't load data without data_part ({data_part}) set.")
        for part in data_part:
            tmp_data = self.load_npz(metadata, part_name=part)
            # log.debug(dict_to_string(tmp_data, part))
            data.update(tmp_data)
            
        temporary_limit = True
        if not temporary_limit:
            return data
        ''' temporary solution for large float number in scene_light'''
        prefs = ['']
        for scale_name in self.buffer_config['scale_regex']:
            scale_config = self.buffer_config['scale_regex'][scale_name]
            if scale_config['enable']:
                prefs.append(scale_config['target'].format(int(scale_config['value'])) + "_")
        if self.gpu:
            data = data_to_device(data, "cuda")
        return data

    def load(self, metadata: MetaData, history_config=None, allow_skip=False):
        '''
        history_config = {
            'num':1,
            'part':['fp16', 'fp32']
            'augmented_data':[]
        }
        '''
        if history_config is not None:
            if history_config['generate_from_data']:
                data = self.load_data(metadata)
                num = history_config['num']
                history_data_list = [{} for _ in range(num)]
                for k in data.keys():
                    res = parse_buffer_name(k)
                    pref = res['sample']
                    buffer_name = res['buffer_name']
                    his_id = res['his_id']
                    postf = res['postf']
                    if buffer_name.startswith("history_") and "warped" not in buffer_name \
                            and his_id is not None and his_id < num:
                        history_data_list[his_id][pref + buffer_name.replace("history_", "") + postf] = data[k]
            else:
                inds = history_config.get("index", range(history_config["num"]))
                history_data_list = list(reversed([self.load_data(
                    metadata.get_offset(-i - 1), data_part=history_config['part'][i])
                    for i in reversed(inds)]))
                data = self.load_data(metadata)

            ''' augmenting history data '''
            if self.with_augment:
                for ind, history_data in enumerate(history_data_list):
                    if len(list(history_data.keys())) <= 0:
                        continue
                    get_augmented_buffer(
                        history_config['augmented_data'][ind],
                        self.buffer_config,
                        history_data,
                        last_data=[],
                        allow_skip=history_config['allow_skip'])
        else:
            data = self.load_data(metadata)
            history_data_list = []
            
        ''' if augmented_data_output list is empty, then generate it. '''
        if self.with_augment:
            if len(self.require_list) > 0 and self.augmented_data_output is None:
                self.augmented_data_output = []
                ''' if data_config isnt empty, we continue generating. '''
                if self.buffer_config is not None:
                    for item in self.require_list:
                        if item not in data.keys():
                            res = parse_buffer_name(item)
                            pref = res['sample']
                            buffer_name = res['buffer_name']
                            his_id = res['his_id']
                            postf = res['postf']
                            ''' temporay exception for history_masked_warped_scene_color_extranet '''
                            if item == "history_masked_warped_scene_color_0_extranet":
                                buffer_name = item
                            elif "extra" in item:
                                continue
                            # log.debug("'{}' '{}' '{}' '{}'".format(item, buffer_name, his_id, pf))
                            if pref + buffer_name + postf in self.buffer_config['augmented_data_recipe'].keys():
                                self.augmented_data_output.append(item)
                            else:
                                raise NotImplementedError("{}({}) is required but not in augmented_data_recipe.keys ({})".format(
                                    pref + buffer_name + postf, item, self.buffer_config['augmented_data_recipe'].keys()))

            get_augmented_buffer(
                self.augmented_data_output,
                self.buffer_config,
                data,
                last_data=history_data_list,
                allow_skip=allow_skip)

            if len(self.require_list) > 0:
                ret = {}
                for k in data.keys():
                    if k in self.require_list:
                        ret[k] = data[k]
                del data
                data = ret
        else:
            data['history_data_list'] = history_data_list
        if self.gpu:
            data_cpu = data_to_device(data, "cpu")
            del data
            torch.cuda.empty_cache()
            gc.collect()
        else:
            data_cpu = data
        return data_cpu

    def get_file_path(self, metadata: MetaData, dir_name: str, postfix=""):
        return "{}/{}/{}/{}.npz".format(
            self.export_path, metadata.scene_name, dir_name, "{}{}".format(metadata.index, postfix))

    def get_buffered_last_data(self, metadata: MetaData, buffer_config, num: int = 1, augmented_list=[]):
        ret = []
        for i in range(num):
            cur_key = str(metadata.get_offset(-i - 1))
            if cur_key not in self.last_data.keys():
                data = self.load_data(metadata.get_offset(-i - 1))
                if buffer_config['dual']:
                    augmented_list = dualize_buffer_list(augmented_list, post_fixes=[
                        '_l', '_r'], exclusion_names=[])
                if len(augmented_list) > 0:
                    get_augmented_buffer(augmented_list,
                                         buffer_config,
                                         data,
                                         [],
                                         allow_skip=False)
                log.debug("{} not found, regenerated.".format(cur_key))
                self.last_data[cur_key] = data
            else:
                data = self.last_data[cur_key]
                if len(augmented_list) > 0:
                    get_augmented_buffer(augmented_list,
                                         buffer_config,
                                         data,
                                         [],
                                         allow_skip=False)
                self.last_data[cur_key] = data
            ret.append(self.last_data[cur_key])
        return ret

    def load_patch(self, metadata):
        return self.load_data(metadata)

    def extend(self, metadata: MetaData, start_cutoff=5):
        if metadata.index < start_cutoff:
            return
        origin_data = self.load_data(metadata)
        self.last_data[str(metadata)] = origin_data
        if metadata.index>start_cutoff and str(metadata.get_offset(-start_cutoff-1)) in self.last_data:
            del self.last_data[str(metadata.get_offset(-start_cutoff-1))]
        extended_patch = {}

        ''' extend process '''
        for part_name in self.buffer_config['addition_part_enable_list']:
            part_config = self.buffer_config['part'][part_name]
            log.debug(f'extending {metadata} {part_name}')
            last_datas = self.get_buffered_last_data(metadata, self.buffer_config,
                                                     num=part_config['num_history'],
                                                     augmented_list=part_config['history_augmented_data'])

            extended_patch[part_name] = get_extend_buffer(origin_data, part_name,
                                                          self.buffer_config,
                                                          start_cutoff=0,
                                                          last_datas=last_datas)
        if metadata.index == 5:
            for k in extended_patch.keys():
                log.info(dict_to_string(extended_patch[k], k))
        for k in extended_patch.keys():
            if k in self.buffer_config['basic_part_enable_list']:
                continue
            path = self.get_file_path(metadata, k)
            log.debug(dict_to_string(extended_patch[k], k, mmm=True))
            create_dir(get_file_component(path)['path'])
            scene_path = os.path.join(get_file_component(path)['path'], "..")
            write_npz(path, extended_patch[k])
            if metadata.index >= 5 and metadata.index <= 10:
                write_text_to_file(scene_path + "/" + "{}.txt".format(k),
                                   json.dumps(list(extended_patch[k].keys()), indent=4).replace(
                    "true", "True").replace("false", "False").replace("null", "None"), "w")
            log.debug(f"write npz: {path}")

    def export_extend_patch_range(self, metadatas, start, end):
        log.debug("start: {}, end: {}".format(start, end))
        for i in tqdm(range(start, end)):
            self.extend(metadatas[i])

    def export_extend_patch(self):
        for s in self.job_config['scene']:
            file_name_list = glob(
                "{}/{}/{}/*.npz".format(self.export_path, s, self.buffer_config["metadata_part"]))
            num = len(file_name_list)
            metadatas = []
            for i in range(0, num):
                metadatas.append(MetaData(s, i))

            ''' single thread '''
            if self.job_config.get('num_thread', 0) <= 0:
                self.export_extend_patch_range(metadatas, 0, len(metadatas))
                continue

            ''' multi thread '''
            n_core = self.job_config['num_thread']
            pool = mp.Pool(processes=n_core)
            thread_part = max(num // n_core + 1, 1)
            try:
                # ins.export_patch_range(scene, start_index, end_index, file_index, idx)
                log.debug("scene:{} n_core:{} thread_part:{}".format(
                    s, n_core, thread_part))

                _ = [pool.apply_async(self.export_extend_patch_range,
                                      (metadatas, i * thread_part,
                                       min((i + 1) * thread_part, num)),
                                      callback=None)

                     for i in range(n_core)]
                pool.close()
            except KeyboardInterrupt:
                pool.terminate()
            finally:
                pool.join()
