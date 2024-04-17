import copy
import json
import os
import torch
import torch.utils.data._utils

from dataloaders.dataset_base import DatasetBase, MetaData
from dataloaders.patch_loader import PatchLoader
from utils.dataset_utils import merge_lr
from utils.buffer_utils import create_flip_data
import numpy as np
from utils.log import get_local_rank, log
from utils.str_utils import dict_to_string

start_offset = 0

def dual_collate_fn(data):
    data = dict(torch.utils.data._utils.collate.default_collate(data))
    for k in data.keys():
        if isinstance(data[k], torch.Tensor) and len(data[k].shape) == 5:
            shape5 = data[k].shape
            data[k] = data[k].reshape(shape5[0] * shape5[1], shape5[2], shape5[3], shape5[4])
    return data


class ShadeNetDataset(DatasetBase):
    def __init__(self, config, dataset_name, metadatas, patch_loader: PatchLoader, mode):
        super().__init__(
            dataset_name, metadatas, mode=mode)
        self.config = config
        self.batch_size = config['train_parameter']['batch_size']
        self.is_block = config['dataset']['is_block']
        self.is_block_part = config['dataset']['is_block_part']
        self.patch_loader = patch_loader

    def __getitem__(self, index) -> dict:
        data = self.patch_loader.load(self.metadatas[index], history_config=self.config['dataset'].get(
            'history_config', None), allow_skip=False)
        assert (self.metadatas[index].index == data['metadata']['index']-start_offset)
        return data


class ShadeNetV5Dataset(ShadeNetDataset):
    def __init__(self, config, dataset_name, metadata, patch_loader, mode):
        super().__init__(config, dataset_name, metadata, patch_loader, mode)

    def __getitem__(self, index) -> list[dict]:
        datas = [self.patch_loader.load(self.metadatas[index].get_offset(i),
                                      history_config=self.config['dataset'].get('history_config', None), allow_skip=False)
                 for i in range(self.config['dataset']['part_size'])]
        for i, item in enumerate(datas):
            assert self.metadatas[index].get_offset(i).index == item['metadata']['index'] - start_offset
        return datas


class ShadeNetV5SeqDataset(ShadeNetV5Dataset):
    def __init__(self, config, dataset_name, metadata: MetaData, patch_loader, mode):
        super().__init__(config, dataset_name, metadata, patch_loader, mode)

    def __getitem__(self, index):
        datas = [self.patch_loader.load(self.metadatas[index].get_offset(i),
                                        history_config=self.config['dataset'].get('history_config', None), allow_skip=False)
                 for i in range(self.config['dataset']['block_size'])]

        if self.mode == "train" and self.config['dataset'].get("flip", True):
            vertical = False
            horizontal = False
            if np.random.random() > 0.5:
                vertical = True
            if np.random.random() > 0.5:
                horizontal = True
                
            for i, data in enumerate(datas):
                assert self.metadatas[index].get_offset(i).index == data['metadata']['index'] - start_offset
                data['metadata']['vertical_flip'] = vertical
                data['metadata']['horizontal_flip'] = horizontal
                if not vertical and not horizontal:
                    continue
                data = create_flip_data(data, vertical=vertical, horizontal=horizontal, use_batch=False)
                if 'history_data_list' in data.keys():
                    history_datas = data['history_data_list']
                    for he_id, he_data in enumerate(history_datas):
                        history_datas[he_id] = create_flip_data(he_data, vertical=vertical, horizontal=horizontal, use_batch=False)
        return datas

