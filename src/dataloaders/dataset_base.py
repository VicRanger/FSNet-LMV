from __future__ import annotations
import time
import multiprocessing as mp
import numpy as np
# from utils.dataset_utils import motion_vector_to_flow
from torch.utils.tensorboard.summary import scalar
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.str_utils import dict_to_string
from utils.buffer_utils import fix_dmdl_color_zero_value, gamma_log, inv_gamma_log, write_buffer
from utils.log import log
import random
import torch
import glob


class MetaData:
    def __init__(self, scene_name, index):
        self.scene_name = scene_name
        self.index = index
        
    def get_offset(self, offset):
        if self.index + offset < 0:
            raise RuntimeError(
                "no offset frame for no.({}+{}={}).".format(self.index, offset, self.index + offset))
        return MetaData(self.scene_name, self.index + offset)

    def __str__(self):
        return "{}_{}".format(self.scene_name, self.index)

    def __repr__(self):
        return str(self.to_dict())

    def to_dict(self):
        return {'scene_name': self.scene_name,
                'index': self.index}
        
    # def previous_of(self, md):
    #     return self.scene_name == md.scene_name


def range_task(task, metadatas, start_idx, end_idx):
    time.sleep(end_idx * 0.001)
    log.debug(f"start range_task[{start_idx}:{end_idx}]")
    for i in tqdm(range(start_idx, end_idx)):
        task(metadatas[i])


def dispatch_task_by_metadata(task, metadatas: list[MetaData], num_thread=0):
    ''' single thread '''
    if num_thread <= 0:
        range_task(task, metadatas, 0, len(metadatas))
        return
    ''' multi thread '''
    n_core = num_thread
    num = len(metadatas)
    pool = mp.Pool(processes=n_core)
    thread_part = max(num // n_core + 1, 1)
    try:
        log.debug("scene:{} n_core:{} thread_part:{}".format(
            metadatas[0].scene_name, n_core, thread_part))
        _ = [pool.apply_async(range_task, (task, metadatas, i * thread_part,
                                           min((i + 1) * thread_part, num), ),
                              callback=None)
             for i in range(n_core)]
        pool.close()
    except KeyboardInterrupt:
        pool.terminate()
    except Exception as e:
        log.debug(e)
        pool.terminate()
    finally:
        pool.join()


def create_metadata_by_glob(path, scene, part_name):
    file_name_list = glob.glob(
        "{}/{}/{}/*.npz".format(path, scene, part_name))
    num = len(file_name_list)
    metadatas = []
    for i in range(0, num):
        metadatas.append(MetaData(scene, i))
    return metadatas


def create_meta_data_list(config, start_cutoff=3):
    shuffle = config['dataset']['shuffle_metadata']
    # shuffle_loader = config['dataset']['shuffle_loader']
    train_list = []
    test_list = []
    valid_list = []
    batch_size = config['train_parameter']['batch_size']
    num_gpu = config['num_gpu']
    path = config['dataset']['path']
    is_block = config['dataset']['is_block']
    is_block_part = config['dataset']['is_block_part']
    # vbs = 1
    if is_block:
        block_size = config['dataset']['block_size']
    else:
        block_size = 0

    if "sep" in config['dataset']['mode']:
        train_scenes = list(config['dataset']['train_scene'])
        for item in train_scenes:
            dir_name = item['name']
            res = glob.glob(f"{path}/{dir_name}/{config['buffer_config']['basic_part_enable_list'][0]}/[0-9]*.npz")
            if len(res) <= 0:
                raise Exception(f"{config['buffer_config']['basic_part_enable_list'][0]} in {path}/{dir_name} not found.")

            sep_rule = item['config'].get('indice', [])
            if len(sep_rule) == 1:
                num = sep_rule[0]
                index = np.arange(start_cutoff, start_cutoff + num)
            elif len(sep_rule) == 2:
                start = sep_rule[0]
                end = sep_rule[1]
                num = end - start
                index = np.arange(start_cutoff + start, start_cutoff + end)
            else:
                num = len(res) - start_cutoff
                index = np.arange(start_cutoff, start_cutoff + num)
            if is_block:
                index = index[:-block_size+1:block_size]
                num = len(index)

            log.debug(dict_to_string([start_cutoff, num, block_size, index]))
            train_list += [MetaData(dir_name, index[i])
                           for i in range(num)]

            log.info("train_scene: {}, path: {} len: {}".format(
                dir_name,
                path,
                num))

        test_scenes = list(config['dataset']['test_scene'])
        for item in test_scenes:
            dir_name = item['name']
            res = glob.glob(f"{path}/{dir_name}/{config['buffer_config']['basic_part_enable_list'][0]}/[0-9]*.npz")
            if len(res) <= 0:
                raise Exception(f"{config['buffer_config']['basic_part_enable_list'][0]} in {path}/{dir_name} not found.")

            num = len(res) - start_cutoff
            index = np.arange(start_cutoff, start_cutoff + num)
            sep_rule = item['config'].get('indice', [])
            if len(sep_rule) == 1:
                end = sep_rule[0]
                index = index[:end]
            elif len(sep_rule) == 2:
                start = sep_rule[0]
                end = sep_rule[1]
                index = index[start:end]

            if is_block and not is_block_part:
                index = index[:-block_size+1:block_size]

            num = len(index)

            test_list += [MetaData(dir_name, index[i])
                          for i in range(num)]
            log.info("test_scene: {}, path: {} len: {}".format(
                dir_name,
                path,
                num))
    else:
        raise NotImplementedError(
            f"create dataset with {config['dataset']['mode']} mode, but only \'seq\' mode supported for dataset!")

    is_initial_shuffle_metadata = True
    if is_initial_shuffle_metadata:
        random.seed(2024)
        random.shuffle(train_list)
        
    if shuffle:
        random.seed(time.time())
        random.shuffle(train_list)
    

    train_scale = config["dataset"].get("train_scale", 1)

    if train_scale != 1:
        log.debug(f"train_scale={train_scale}, scaling train_list(len={len(train_list)})")
        np.random.seed(2024)
        train_ind = np.random.choice(np.arange(len(train_list), dtype=int), int(len(train_list) * train_scale), replace=False)
        train_list = list(np.array(train_list)[train_ind])
        log.debug(f"scaled train_list(len={len(train_list)})")
    
    if is_block:
        minimum_total_size = num_gpu * batch_size
        while len(train_list) % (minimum_total_size) != 0:
            train_list += train_list[:minimum_total_size - len(train_list) % minimum_total_size]
            
    
    if is_block_part:
        def generate_block_metadata(block_list: list[MetaData], _batch_size, _num_gpu, _block_size, round_block=False):
            part_size = config['dataset']['part_size']
            assert _block_size % part_size == 0 
            _minimum_total_size = _num_gpu * _batch_size
            expand_list = []
            for md in block_list:
                expand_list.append(md)
                for block_id in range(part_size, _block_size, part_size):
                    expand_list.append(md.get_offset(block_id))
            len_expand_list = len(expand_list) 
            parted_block_size = _block_size // part_size
            log.debug(f'len_block_list={len(block_list)} len_expand_list={len_expand_list} num_block={len(block_list) // _minimum_total_size * _minimum_total_size * parted_block_size}')
            assert len_expand_list == len(block_list) // _minimum_total_size * _minimum_total_size * parted_block_size
            ret_list = []
            len_batched_seq = _batch_size * parted_block_size
            for seq_id in range(len_expand_list // len_batched_seq):
                cut_list = expand_list[seq_id * len_batched_seq: (seq_id+1) * len_batched_seq]
                for block_id in range(parted_block_size):
                    for batch_id in range(0, _batch_size):
                        ret_list.append(cut_list[batch_id * parted_block_size + block_id])
            return ret_list
        train_list = generate_block_metadata(train_list, batch_size, num_gpu, block_size)
        # valid_list = generate_block_metadata(valid_list, 1, 1, block_size)
        # test_list = generate_block_metadata(test_list, 1, 1, block_size)

    log.debug("train: {} ... {}".format(str(train_list[:3]), str(train_list[-3:])))
    log.debug("test: {} ... {}".format(str(test_list[:3]), str(test_list[-3:])))
    log.info("complete creating metadata.")
    return train_list, valid_list, test_list


''' a single frame meta info, not including data. '''
''' in raw_data_importer, index is the frame index. '''
''' in patch_loader, index is the npz index, (= frame_index - start_offset) '''


class CropMetaData(MetaData):
    def __init__(self, scene_name, index, global_index, skybox_ratio, discontinuity_ratio):
        super().__init__(scene_name, index)
        # index is frame_index, global_index is cropped patch index
        self.global_index = global_index
        self.skybox_ratio = skybox_ratio
        self.discontinuity_ratio = discontinuity_ratio

    def to_dict(self):
        return {'scene_name': self.scene_name,
                'index': self.index,
                'global_index': self.global_index,
                'skybox_ratio': self.skybox_ratio,
                'discontinuity_ratio': self.discontinuity_ratio}


class DatasetBase(Dataset):
    def __init__(self, dataset_name, metadatas: list[MetaData], mode="train"):
        self.dataset_name = dataset_name
        self.metadatas = metadatas
        self.mode = mode
        log.info("dataset_name: {}, data_size: {}".format(
            self.dataset_name, self.__len__()))

    @staticmethod
    def preprocess(data, config={}):
        ret = {}
        for name in data.keys():
            # if ('world_position' in name):
            #     scale_factor = 2.0
            #     data[name] = torch.clamp(
            #         data[name], -65536.0 * scale_factor, 65536.0 * scale_factor)
            #     data[name] = data[name] / 65536.0 / scale_factor
            if ('scene_light' in name or 'scene_color' in name or 'sky_color' in name or 'st_color' in name):
                ret[name] = gamma_log(data[name], mu=config.get('mu', 8.0))
            elif 'normal' in name:
                ret[name] = data[name] * 0.5 + 0.5
            else:
                ret[name] = data[name]
        return ret

    @staticmethod
    def postprocess(data):
        for name in data.keys():
            if ('scene_light' in name or 'scene_color' in name or 'sky_color' in name or 'st_color' in name):
                data[name] = inv_gamma_log(data[name])

    def __len__(self) -> int:
        return len(self.metadatas)

    def __getitem__(self, index) -> dict:
        return {}
