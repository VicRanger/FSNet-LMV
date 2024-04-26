from __future__ import annotations
import copy
import random
import torch
from torch.amp.autocast_mode import autocast as autocast
from torch.cuda.amp.grad_scaler import GradScaler
import torch.utils.data.distributed
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
from samplers.distributed_partition_sampler import DistributedPartitionSampler
from dataloaders.dataset_base import create_meta_data_list
from dataloaders.patch_loader import PatchLoader
from utils.loss_utils import lpips, psnr, psnr_hdr, ssim, ssim_hdr
from utils.utils import add_at_dict_front, create_dir, get_file_component, get_tensor_mean_min_max_str, \
    remove_all_in_dir, write_text_to_file
from utils.str_utils import dict_to_string, dict_to_string_join
from utils.log import log
from utils.warp import warp
from utils.dataset_utils import data_to_device, get_input_filter_list, resize
from utils.utils import del_data, del_dict_item
from utils.buffer_utils import buffer_data_to_vis, gamma, to_numpy, write_buffer
from models.loss.loss import LossFunction
from models.model_base import ModelBase
from tqdm import tqdm
from utils.buffer_utils import align_channel_buffer
import itertools
import torch.autograd
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader
from datetime import datetime, timedelta, timezone
from glob import glob
import json
import os
import re
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as F
import time
import math
from config.config_utils import convert_to_dict
from datasets.shade_net_dataset import ShadeNetDataset, ShadeNetV5Dataset


def should_active_at(step, total_step, total: int, offset=0, last=-1e9) -> bool:
    if step+offset - last < max(1, total_step // total):
        return False
    return (step - offset) % (max(1, total_step // total)) == 0


def get_time_string_in_dir(path):
    dirs = glob(path + "/*")
    dirs = [str(d).replace('\\', '/') for d in dirs]
    log.debug(dirs)
    newest_stamp = ""
    find = []
    for d in dirs:
        res = re.search(
            r"(.+)/([\d]+)-([\d]+)-([\d]+)_([\d]+)-([\d]+)-([\d]+)", str(d))
        if (res):
            tmp_stamp = "{}-{}-{}_{}-{}-{}".format(res.group(2), res.group(
                3), res.group(4), res.group(5), res.group(6), res.group(7))
            if tmp_stamp > newest_stamp:
                find.append(tmp_stamp)
                newest_stamp = tmp_stamp
    if len(newest_stamp) > 0:
        log.debug("find newest_stamp \"{}\" in {}".format(newest_stamp, dirs))
    else:
        raise ValueError(f'cant find proper newest_stamp. possibly there is no train result under "{[path]}"')
    return newest_stamp


step_log_interval = 4
step_print_interval = 10


class TrainerBase:
    def __init__(self, config, model: ModelBase, resume=False):
        config['_trainer_config'] = copy.deepcopy(convert_to_dict(config))
        self.resume = resume
        self.config = config
        self.class_name = self.config['trainer']['class']
        self.step_log_interval = self.config['log'].get('step_log_interval', step_log_interval)
        self.step_print_interval = self.config['log'].get('step_print_interval', step_print_interval)
        self.model = model
        if self.model is not None:
            self.net = self.model.get_net()
        self.train_loader = None
        self.test_loader = None
        self.valid_loader = None
        self.scaler = None
        self.optimizer = None
        self.scheduler = None
        self.writer = None
        self.next_write_debug_info_step_offset = 1
        self.last_write_debug_info_step = -1e9
        self.local_rank = int(self.config['local_rank'])
        self.enable_log = self.local_rank <= 0
        self.enable_test = self.local_rank <= 0

        self.shuffle_loader = self.config['dataset']['shuffle_loader']
        self.dataset_scaled = self.config['dataset'].get('scale', 1) != 1
        self.train_config = config['train_parameter']
        self.job_name = config['job_name']
        self.output_path = config['output_root_path'] + self.job_name
        self.debug_data_flow = config['trainer']['debug_data_flow']

        if self.resume:
            if 'time_string' in config.keys():
                self.time_string = config['time_string']
            else:
                self.time_string = get_time_string_in_dir(
                    "{}/".format(self.output_path))
                log.debug(self.time_string)
        else:
            gmt_format = "%Y-%m-%d_%H-%M-%S"
            tz = timezone(timedelta(hours=+8))
            self.time_string = datetime.now(tz).strftime(gmt_format)
            self.config['time_string'] = self.time_string
            if self.config.get('clear_output_path', False) and self.enable_log:
                remove_all_in_dir(self.output_path)
        self.log_path = "{}/{}/log/".format(
            self.output_path, self.time_string)
        self.model_path = "{}/{}/model/".format(
            self.output_path, self.time_string)
        self.writer_path = "{}/{}/writer".format(
            self.output_path, self.time_string)
        self.checkpoint_path = "{}/{}/checkpoint".format(
            self.output_path, self.time_string)
        self.history_checkpoint_path = "{}/{}/history_checkpoints".format(
            self.output_path, self.time_string)

        self.batch_size = self.train_config['batch_size']
        self.lr = self.train_config['lr']
        self.use_cuda = self.config['use_cuda']
        self.start_epoch = self.train_config.get('start_epoch', 0)
        self.epoch_loop = None
        self.main_loop = None
        self.end_epoch: int = self.train_config['epoch']
        self.total_epoch = self.train_config['epoch']
        self.config['tensorboard_info_step'] = {'train': [], 'test': []}
        self.config['tensorboard_info_epoch'] = {'train': [], 'test': []}
        self.config['bar_info_step'] = {'train': [], 'test': []}
        self.config['bar_info_epoch'] = {'train': [], 'test': []}
        self.config['text_info_step'] = {'train': [], 'test': []}
        self.config['text_info_epoch'] = {'train': [], 'test': []}
        self.config['avg_info_epoch'] = {'train': [], 'test': []}
        for mode in ['train', 'test']:
            for source in self.config['log'][mode].keys():
                item = self.config['log'][mode][source]
                item['name'] = item.get('name', source)
                if item.get('log_step', False):
                    self.config['tensorboard_info_step'][mode].append(source)
                if item.get('log_epoch', False):
                    self.config['tensorboard_info_epoch'][mode].append(source)
                if item.get('bar_step', False):
                    self.config['bar_info_step'][mode].append(source)
                if item.get('bar_epoch', False):
                    self.config['bar_info_epoch'][mode].append(source)
                if item.get('text_step', False):
                    self.config['text_info_step'][mode].append(source)
                if item.get('text_epoch', False):
                    self.config['text_info_epoch'][mode].append(source)
                if item.get('bar_epoch', False) or item.get('log_epoch', False) or item.get('text_epoch', False):
                    self.config['avg_info_epoch'][mode].append(source)

        self.epoch_index = 0
        self.save_best_epoch = max(int(self.train_config.get('save_best_at', 0) * self.end_epoch), 1) - 1
        self.batch_index = 0
        self.step = 0
        self.step_per_epoch = 0
        self.total_step = 0
        self.resume_step = -1
        self.timestamp = time.time()
        self.start_time = time.time()
        self.min_loss = 1e9
        self.latent_step = 0
        self.best_step = -1
        self.data_time_interval = 0
        self.infer_time_interval = 0
        self.loss = data_to_device(torch.tensor(
            0.0), device=self.config['device'])
        self.loss_func = None
        self.active_loss_funcs = []

        self.info_epoch_avg = {}
        self.info_count = {}
        self.log_image_step_count = 0
        self.log_scalar_step_count = 0

        self.cur_data = {}
        self.cur_output = {}
        self.cur_lr = 0.0
        self.cur_loss = {}
        self.cur_loss_debug = {}
        self.enable_amp = self.train_config.get('amp', False)
        # self.prepare()

    def create_dataset(self) -> None:
        if not self.config['dataset'].get('enable', True):
            return
        self.train_meta_data, self.valid_meta_data, self.test_meta_data = create_meta_data_list(
            self.config)
        if len(self.train_meta_data) <= 0:
            raise RuntimeError("train dataset not found.")
        # calc input filter
        require_list = self.config['dataset']['require_list']
        # log.debug(require_filter)

        self.data_loader = PatchLoader(
            self.config['dataset']['part'],
            job_config={'export_path': self.config['dataset']['path']},
            buffer_config=self.config['buffer_config'],
            require_list=require_list,
            with_augment=self.config['dataset']['augment_loader']
        )
        self.train_dataset = eval(self.config['dataset']['class'])(
            self.config, 'train', self.train_meta_data, self.data_loader, mode="train")

        test_config = copy.deepcopy(self.config)
        # test_config['buffer_config']['crop_config']['enable'] = False
        self.test_dataset = eval(self.config['dataset']['class'])(
            test_config, 'test', self.test_meta_data, self.data_loader, mode="test")

        if len(self.valid_meta_data) > 0:
            self.valid_dataset = eval(self.config['dataset']['class'])(
                test_config, 'valid', self.valid_meta_data, self.data_loader, mode="valid")
        else:
            self.valid_dataset = None

    def create_loader(self) -> None:
        if not self.config['dataset'].get('enable', True):
            return
        pin_memory = self.config['dataset']['pin_memory']
        if self.config['use_ddp']:
            torch.utils.data.distributed.DistributedSampler
            if self.config['dataset']['is_block'] and self.config['dataset']['is_block_part']:
                self.train_sampler = DistributedPartitionSampler(self.train_dataset,
                                                                                    num_replicas=self.config['world_size'],
                                                                                    rank=self.config['local_rank'],
                                                                                    shuffle=self.config['dataset']['shuffle_loader'])
            else:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset,
                                                                                    num_replicas=self.config['world_size'],
                                                                                    rank=self.config['local_rank'],
                                                                                    shuffle=self.config['dataset']['shuffle_loader'])
        else:
            self.train_sampler = None
        collate_fn = None
        if self.config['buffer_config']['dual']:
            collate_fn = dual_collate_fn
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size,
                                       num_workers=self.config['dataset']['train_num_worker'],
                                       pin_memory=pin_memory,
                                       drop_last=True,
                                       sampler=self.train_sampler,
                                       shuffle=self.shuffle_loader,
                                       collate_fn=collate_fn)
        #    shuffle=self.config['dataset']['shuffle_loader'])
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=1,
                                      num_workers=self.config['dataset']['test_num_worker'],
                                      pin_memory=False,
                                      shuffle=False,
                                      collate_fn=collate_fn)
        if self.valid_dataset:
            self.valid_loader = DataLoader(self.valid_dataset,
                                           batch_size=1,
                                           num_workers=self.config['dataset']['test_num_worker'],
                                           pin_memory=False,
                                           shuffle=self.shuffle_loader)

    def create_scaler(self):
        self.scaler = GradScaler(enabled=self.enable_amp)

    def create_optimizer(self):
        initial_lr = self.train_config['lr'] * self.train_config['batch_size'] * self.config.get("num_gpu", 1)
        betas = tuple(self.train_config.get('betas', (0.9, 0.999)))
        log.debug(betas)
        self.optimizer = AdamW(
            itertools.chain(self.model.net.parameters()),  # type: ignore
            lr=initial_lr,
            betas=betas,
            weight_decay=1e-4)
        # self.optimizer = Adam(
        #     itertools.chain(self.model.net.parameters()),
        #     lr=1e-6,
        # )
        if self.train_config['lr_mode'] == "cosine_annealing_warm_restarts":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=int(1 / self.train_config['lr_config']['epoch'] * self.end_epoch * self.step_per_epoch),
                eta_min=self.train_config['lr_config']['eta_min'],
                T_mult=1,
            )

    def create_loss_func(self) -> None:
        self.loss_func = eval(self.config["loss"]['class'])(
            self.config["loss"])
        self.cur_data = copy.deepcopy(self.model.dummy_net_input)
        self.model.set_eval()
        self.cur_output = self.model.forward(self.cur_data)
        self.gather_execute_result(training=True, enable_loss=True)
        tmp_output = self.cur_output
        tmp_output.update(self.cur_data)
        self.loss_func.check_data(tmp_output)
        self.active_loss_funcs = self.loss_func.get_active_loss_func_names()
        self.cur_data.clear()
        self.cur_output.clear()
        log.debug("active loss_func name: {}".format(self.active_loss_funcs))
        
        
    def set_step_per_epoch(self, mode):
        if mode == 'train':
            if self.train_loader:
                self.step_per_epoch = self.train_loader.__len__()
            else:
                self.step_per_epoch = 0
            if self.config['dataset']['is_block']:
                if self.config['dataset']['is_block_part']:
                    self.step_per_epoch *= self.config['dataset']['part_size'] 
                else:
                    self.step_per_epoch *= self.config['dataset']['block_size']
        elif mode == 'test':
            if self.test_loader:
                self.step_per_epoch = self.test_loader.__len__()
                if self.config['dataset']['is_block']:
                    if not self.config['dataset']['is_block_part']:
                        self.step_per_epoch *= self.config['dataset']['block_size']
        else:
            assert False
        

    def prepare(self, mode='train') -> None:
        if self.config['dataset'].get('enable', True):
            self.create_dataset()
            self.create_loader()
            log.info("[{}]: dataset created.".format(
                self.config['trainer']['class']))

        if mode == 'train':
            self.set_step_per_epoch(mode)
            self.total_step = self.step_per_epoch * \
                (self.end_epoch)
            self.resume_step = self.step_per_epoch * \
                (self.end_epoch - self.start_epoch)
            self.latent_step = 20 * self.step_per_epoch
            self.create_scaler()
            self.create_optimizer()

        elif mode == 'test':
            self.set_step_per_epoch(mode)
            self.total_step = self.step_per_epoch

        if self.model is not None:
            self.create_loss_func()

        if self.resume:
            self.load_checkpoint()

        if self.model is not None and not self.resume and "pre_model" in self.config.keys():
            self.model.load_by_path(self.config["pre_model"])
            log.debug("pre_model loaded: {}".format(self.config["pre_model"]))

        if self.config['use_ddp']:
            log.debug(f"model to_ddp, local_rank:{self.config['local_rank']}")
            self.model.to_ddp(
                device_ids=[self.config['local_rank']],
                output_device=self.config['local_rank'])

        if self.model is not None and self.enable_log:
            create_dir(self.log_path)
            create_dir(self.model_path)
            create_dir(self.writer_path)
            self.writer = SummaryWriter(self.writer_path)
            if not self.resume:
                config_file_path = self.log_path + \
                    "input_{}.json".format("{}_{}".format(
                        self.config['job_name'].replace("/", "_"), "runtime"))

                write_text_to_file(config_file_path, json.dumps(self.config['_input_config'], indent=4), "w")

                config_file_path = self.log_path + \
                    "trainer_{}.json".format("{}_{}".format(
                        self.config['job_name'].replace("/", "_"), "runtime"))

                write_text_to_file(config_file_path, json.dumps(self.config['_trainer_config'], indent=4), "w")

                self.write_text_info(
                    text_dict_to_tensorboard=self.get_info_dict('tensorboard', 'initial'),
                    text_dict_to_file=self.get_info_dict('text', 'initial'),
                    log_name="info",
                    enable_log=True
                )
                
            step_info = {
                'step': self.step,
                'resume_step': self.resume_step,
                'step_per_epoch': self.step_per_epoch,
                'total_step': self.total_step,
                'total_epoch': self.total_epoch,
                'step_log_interval': self.step_log_interval,
                'step_print_interval': self.step_print_interval,
            }
            self.write_text_info(
                text_dict_to_tensorboard=None,
                text_dict_to_file=step_info,
                log_name="log",
                enable_log=True
            )
            log.debug(dict_to_string(step_info))

    def load_data(self, data):
        if self.use_cuda:
            self.cur_data: dict = data_to_device(data, self.config['device'], non_blocking=True)  # type: ignore
        if not self.config['dataset']['augment_loader']:
            self.cur_data = self.model.get_augment_data(self.cur_data)

    def update_forward(self, epoch_index=None, batch_index=None, mode="train"):
        # log.debug("{} forward".format(self.config['job_name']))
        if epoch_index is not None:
            self.epoch_index = epoch_index
        if batch_index is not None:
            self.batch_index = batch_index
        # log.debug(dict_to_string(self.cur_data, mmm=True))
        if mode == "test":
            self.before_infer()
            self.execute_model(training=False)
        if mode == "train":
            with autocast(device_type="cuda", dtype=torch.float16, enabled=self.enable_amp):
                self.execute_model(training=True)

    def update_backward(self, epoch_index=None, batch_index=None, mode="train"):
        if epoch_index is not None:
            self.epoch_index = epoch_index
        if batch_index is not None:
            self.batch_index = batch_index
        if mode == "test":
            self.after_infer()
        with autocast(device_type="cuda", dtype=torch.float16, enabled=self.enable_amp):
            self.calc_loss_func(mode)
        if mode == "train":
            self.backward()
        self.update_step(mode)
        self.step += 1

    def update(self, data, epoch_index=None, batch_index=None, mode="train"):
        self.load_data(data)
        self.update_forward(epoch_index=epoch_index,
                            batch_index=batch_index, mode=mode)
        if mode == "train":
            self.gather_execute_result(training=True, enable_loss=True)
        if mode == "test":
            self.gather_execute_result(enable_loss=True)
        self.update_backward(epoch_index=epoch_index,
                             batch_index=batch_index, mode=mode)

    def get_epoch_description_str(self, mode):
        if mode == 'train':
            return "{}train:{}/{},(BS{}/NW{}/NDV{}\"{}\"),S{}/I{}".format(
                "RK:{} ".format(
                    self.config['local_rank']) if self.config['local_rank'] >= 0 else "",
                self.step,
                self.total_step,
                self.batch_size,
                self.config['dataset']['train_num_worker'],
                self.config['num_gpu'],
                os.environ.get(
                    'CUDA_VISIBLE_DEVICES', ""),
                self.log_scalar_step_count,
                self.log_image_step_count)
        elif mode == 'test':
            return "[{}] {}test:{}/{},(NDV{}\"{}\"),S{}/I{}".format(
                self.config['job_name'],
                "RK:{} ".format(
                    self.config['local_rank']) if self.config['local_rank'] >= 0 else "",
                self.step,
                self.total_step,
                self.config['num_gpu'],
                os.environ.get(
                    'CUDA_VISIBLE_DEVICES', ""),
                self.log_scalar_step_count,
                self.log_image_step_count)

    def train(self):
        if self.start_epoch >= self.end_epoch:
            return
        self.prepare('train')
        # torch.autograd.set_detect_anomaly(True)
        log.info("start training, start_epoch:{}, end_epoch:{}".format(
            self.start_epoch, self.end_epoch))
        with tqdm(range(self.start_epoch, self.end_epoch),
                  position=0, leave=True, disable=not (self.enable_log)) as self.main_loop:
            for self.epoch_index in self.main_loop:
                # self.train_loader.sampler.set_epoch(self.epoch_index)
                self.reset_info_accumulator("train")
                self.main_loop.set_description_str(
                    "[{}] epoch: {}".format(self.config['job_name'], self.epoch_index))
                # with tqdm(range(self.train_loader.__len__()), position=1, leave=True) as test_epoch_loop:
                #     for _ in test_epoch_loop:
                #         log.debug("epoch: {} in-epoch step:{}".format(self.epoch_index, _))
                if self.dataset_scaled:
                    self.create_dataset()
                    self.create_loader()
                with tqdm(self.train_loader, position=1, leave=True, disable=not (self.enable_log)) as self.epoch_loop:
                    for self.batch_index, data in enumerate(self.epoch_loop):
                        # log.debug(dict_to_string(data, mmm=True))
                        if self.enable_log:
                            # bs: batch_size, nw: num_worker, dv: device
                            self.epoch_loop.set_description_str(self.get_epoch_description_str('train'))
                        self.update(data, mode="train")
                        self.cur_data.clear()
                        self.cur_loss.clear()
                        self.cur_loss_debug.clear()
                        # torch.cuda.empty_cache()
                        # gc.collect()
                        # log.debug(dict_to_string(self.info_epoch_avg))
                self.update_epoch(mode="train")
                self.cur_output.clear()
                torch.cuda.empty_cache()
                gc.collect()
                self.create_dataset()
                self.create_loader()

    def test(self):
        self.prepare(mode='test')
        ''' reset self.step to 0, in case of mis-modification in self.prepare('test') '''
        self.step = 0
        log.debug("test start")
        if not (self.enable_test):
            return
        if not ("pre_model" in self.config.keys()):
            self.load_model(name="best")
        self.log_scalar_step_count = self.log_image_step_count = 0
        self.reset_info_accumulator("test")
        with tqdm(self.test_loader, position=0, leave=True) as self.epoch_loop:
            for self.batch_index, data in enumerate(self.epoch_loop):
                if self.enable_log:
                    # bs: batch_size, nw: num_worker, dv: device
                    self.epoch_loop.set_description_str(self.get_epoch_description_str('test'))
                self.update(data, mode="test")
                self.cur_data.clear()
                self.cur_loss.clear()
                self.cur_loss_debug.clear()
                epoch_info_bar = self.get_info_dict('bar', 'step', 'test')
                log.info("test epoch: {}".format(epoch_info_bar))
        self.update_epoch("test")
        self.cur_output.clear()
        torch.cuda.empty_cache()

    def load_model(self, path=None, name="best"):
        if path is None:
            path = "{}/{}.pt".format(
                self.model_path, name)
        self.model.load_by_path(path)
        log.info("loaded model: {}".format(path))

    def save_model(self, file_name="model"):
        file_path = self.model_path + "{}.pt".format(file_name)
        info_path = self.model_path + "{}.log".format(file_name)
        self.model.save(file_path)
        f = open(info_path, "w")
        loss = self.get_avg_info("loss")
        if loss == 0:
            loss = self.get_info('loss')
        f.write("epoch: {}, step: {} loss: {:.6f} last_update_step: {}".format(
            self.epoch_index,
            self.step,
            loss,
            self.best_step))
        # FIXME: dist.barrier will lead to program freezed and cuase nccl timeout
        # if self.config['use_ddp']:
        #     dist.barrier()
        log.info("saved model: {}.".format(file_path))

    def save_checkpoint(self, state_dict=True):
        if create_dir(self.checkpoint_path):
            log.info("generate checkpoint save_dir: \"{}\".".format(
                self.checkpoint_path))
        if create_dir(self.history_checkpoint_path):
            log.info("generate history checkpoint save_dir: \"{}\".".format(
                self.history_checkpoint_path))
        if remove_all_in_dir(self.checkpoint_path):
            log.debug("remove all files in save_dir: \"{}\".".format(
                self.checkpoint_path))
        checkpoint_dict = {
            'step': self.step,
            'epoch': self.epoch_index,
            'loss': self.get_model_loss(),
            'optimizer': self.optimizer,
            'scaler': self.scaler,
            'model': self.model.get_net(),
            'best_step': self.best_step,
        }
        torch.save(checkpoint_dict, "{}/checkpoint_{}_{}.pt".format(self.checkpoint_path, self.epoch_index, self.step))
        torch.save(checkpoint_dict, "{}/checkpoint_{}_{}.pt".format(self.history_checkpoint_path, self.epoch_index, self.step))
        log.info("saved checkpoint: {}.".format(self.checkpoint_path))

    def load_checkpoint(self) -> None:
        path_template = "{}/checkpoint_{{}}_{{}}.pt".format(
            self.checkpoint_path)
        files = glob(path_template.format("[0-9]*", "[0-9]*"))
        if len(files) != 1:
            raise FileNotFoundError(
                "file ({}) of {} is not satisfied: {}".format(path_template, self.job_name, files))
        file_path = files[-1]
        file_res = get_file_component(file_path)
        res = re.match("(.*)[_](.*)[_](.*)", file_res['filename'])
        if res:
            epoch = res.groups()[1]
            step = res.groups()[2]
            load_path = path_template.format(epoch, step)
            checkpoint = torch.load(load_path)
            self.step = checkpoint['step']
            self.start_epoch = checkpoint['epoch'] + 1
            tmp_optimizer = checkpoint['optimizer']
            tmp_scaler = checkpoint['scaler']
            tmp_model = checkpoint['model']
            if self.optimizer is not None:
                self.optimizer.load_state_dict(tmp_optimizer.state_dict())
            if self.scaler is not None:
                self.scaler.load_state_dict(tmp_scaler.state_dict())
            self.model.load_by_dict(tmp_model.state_dict())
            log.info("loaded checkpoint {}. step:{}, epoch:{}, loss:{}".format(
                self.model.model_name, checkpoint['step'], checkpoint['epoch'], checkpoint['loss']))
        else:
            raise Exception(f'wrong name of checkpoint file_name, please check path "{file_path}"')

    def before_infer(self) -> None:
        self.data_time_interval = time.time() - self.timestamp
        self.timestamp = time.time()

    def after_infer(self) -> None:
        torch.cuda.synchronize()
        self.infer_time_interval = time.time() - self.timestamp
        self.timestamp = time.time()

    def reset_info_accumulator(self, mode="train") -> None:
        if not self.enable_log:
            return
        config = self.config["avg_info_epoch"][mode]
        for scalar_name in config:
            self.info_epoch_avg[scalar_name] = 0
            self.info_count[scalar_name] = 0

    def step_info_accumulator(self, mode="train") -> None:
        # log.debug(self.config["avg_info_epoch"])
        config = self.config["avg_info_epoch"][mode]
        for scalar_name in config:
            info = self.get_info(scalar_name)
            if info is None or torch.isnan(torch.tensor(info)) or torch.isinf(torch.tensor(info)):
                pass
            else:
                self.info_epoch_avg[scalar_name] += info
                self.info_count[scalar_name] += 1

    def get_avg_info(self, name) -> float | None:
        if name not in self.info_count.keys() or self.info_count[name] == 0:
            return None
        return self.info_epoch_avg[name] / self.info_count[name] if self.info_count[name] else 0

    def update_step(self, mode="train"):
        if not self.enable_log:
            return
        # self.log_data_step()
        self.write_tensorboard_step(mode=mode)
        enable_step_log = True
        enable_step_print = True
        if mode == "train":
            enable_step_log = (self.step % self.step_log_interval == 0)
            enable_step_print = (self.step % self.step_print_interval == 0)
        if enable_step_print and self.debug_data_flow:
            self.log_data_step()
        if mode == "test":
            enable_step_log = True
        # step_info_bar = self.get_step_bar(mode=mode)
        step_info_bar = self.get_info_dict('bar', 'step', trainer_mode=mode)
        if self.epoch_loop is not None:
            self.epoch_loop.set_postfix(step_info_bar)
        if enable_step_log:
            step_info_text = {'step': self.step}
            step_info_text.update(self.get_info_dict('text', 'step', trainer_mode=mode))
            self.write_text_info(
                text_dict_to_file=step_info_text,
                file_line_mid=", ",
                log_name="{}_step".format(mode),
            )
        self.step_info_accumulator(mode=mode)
        
    def get_model_loss(self):
        return self.get_avg_info("loss")

    def update_epoch(self, mode="train"):
        if not self.enable_log:
            return
        if mode == "train":
            if self.config['local_rank'] <= 0:
                loss = self.get_model_loss()
                assert loss is not None
                if loss is not None:
                    if self.epoch_index >= self.save_best_epoch and self.min_loss > loss:
                        self.min_loss = loss
                        self.best_step = self.step
                        self.save_model("best")
                else:
                    raise Exception(f'no key="loss" found in info count, "{self.info_count.keys()}"')

                self.save_model("new")
                self.save_checkpoint()

            # epoch_info_bar = self.get_epoch_bar(mode='train')
            epoch_info_bar = self.get_info_dict('bar', 'epoch', trainer_mode=mode)
            epoch_info_bar = add_at_dict_front(
                epoch_info_bar, "epoch", self.epoch_index)
            if self.main_loop is not None:
                self.main_loop.set_postfix(epoch_info_bar)
            epoch_info_text = {'epoch': self.epoch_index}
            epoch_info_text.update(self.get_info_dict('text', 'epoch', trainer_mode=mode))
            self.write_text_info(
                text_dict_to_file=epoch_info_text,
                file_line_mid=", ",
                log_name="{}_epoch".format(mode),
                enable_log=True
            )
        if mode == "test":
            epoch_info_bar = self.get_info_dict('bar', 'epoch', trainer_mode=mode)
            epoch_info_text = self.get_info_dict('text', 'epoch', trainer_mode=mode)
            log.info("test epoch: {}".format(epoch_info_text))
            self.write_text_info(
                text_dict_to_tensorboard=self.get_info_dict('tensorboard', 'epoch', trainer_mode=mode),
                text_dict_to_file=self.get_info_dict('text', 'epoch', trainer_mode=mode),
                log_name="test",
                enable_log=True
            )
        self.write_to_tensorboard_scalar(step_mode='epoch', mode=mode)

    def _get_float_in_cur_loss(self, name):
        data = self.cur_loss[name]
        if isinstance(self.cur_loss[name], torch.Tensor):
            data = self.cur_loss[name].item()
            self.cur_loss[name] = data
        return data

    def get_info(self, name):
        if name == 'lr':
            if self.scheduler is not None:
                if self.optimizer is not None:
                    return self.optimizer.param_groups[0]['lr']
                else:
                    raise Exception('self.optmizer is None')
            return self.cur_lr
        elif self.cur_loss is not None and name in self.cur_loss.keys():
            # log.debug(f"get_info with cur_loss, key={name}")
            return self._get_float_in_cur_loss(name)
        else:
            return None

    # def get_step_bar(self, mode='train') -> dict:
    #     ret = {}
    #     for source in self.config['bar_info_step'][mode]:
    #         value = self.get_info(source)
    #         if value is None:
    #             continue
    #         name = self.config['log'][mode][source]['name']
    #         fmt = self.config['log'][mode][source]['fmt']
    #         ret[name] = fmt.format(self.get_info(source))
    #     if mode == "test":
    #         ret['infer_time'] = "{:.3g}ms".format(
    #             self.infer_time_interval * 1000)
    #     return ret

    # def get_epoch_bar(self, mode='train') -> dict:
    #     ret = {
    #         "LUS": "{:d}".format(self.best_step)
    #     }
    #     for source in self.config['bar_info_epoch'][mode]:
    #         info = self.get_avg_info(source)
    #         if info is None:
    #             continue
    #         name = self.config['log'][mode][source]['name']
    #         fmt = self.config['log'][mode][source]['fmt']
    #         ret[name] = fmt.format(info)
    #     return ret

    def get_info_dict(self, info_mode, step_mode, trainer_mode='train') -> dict:
        '''
        info_mode: 'bar', 'text', 'tensorboard',
        step_mode: 'step', 'epoch', 'initial'
        trainer_mode: 'train', 'test'
        '''
        ret = {}
        if step_mode == 'initial':
            assert info_mode == 'text' or info_mode == 'tensorboard'
            ret['batch_size'] = self.batch_size
            if self.model is not None:
                ret['infer_time'] = self.model.get_infer_time()
                ret['net_parameter_num'] = self.model.get_net_parameter_num()
            if info_mode == 'text':
                ret['active_loss'] = self.active_loss_funcs
                if self.model is not None:
                    ret['net_struct'] = self.model.get_net()
            return ret
        if info_mode == 'bar':
            if trainer_mode == 'train':
                ret['LUS'] = "{:d}".format(self.best_step)
        print_list = self.config[f'{info_mode}_info_{step_mode}'][trainer_mode]
        if info_mode == 'text':
            for _src in self.config[f'bar_info_{step_mode}'][trainer_mode]:
                if _src not in print_list:
                    print_list.append(_src)
            for _src in self.config[f'tensorboard_info_{step_mode}'][trainer_mode]:
                if _src not in print_list:
                    print_list.append(_src)
        for source in print_list:
            if step_mode == 'step':
                value = self.get_info(source)
            elif step_mode == 'epoch':
                value = self.get_avg_info(source)
            else:
                raise Exception(f'with wrong step_mode: {step_mode}, only "step" and "epoch" supported')
            if value is None:
                continue
            name = self.config['log'][trainer_mode][source]['name']
            fmt = self.config['log'][trainer_mode][source]['fmt']
            ret[name] = fmt.format(value)
        if info_mode == 'bar':
            if trainer_mode == "test":
                ret['infer_time'] = "{:.3g}ms".format(
                    self.infer_time_interval * 1000)
        return ret

    def gather_execute_result(self, training=False, enable_loss=False):
        if enable_loss:
            if self.loss_func is None:
                raise Exception('self.loss_func is None')
            if training:
                losses = self.loss_func.loss_func + self.loss_func.debug_loss_func
            else:
                losses = self.loss_func.debug_loss_func
            for item in losses:
                if not item['enable']:
                    continue
                for arg in item['args']:
                    if arg not in self.cur_output.keys() and arg in self.cur_data.keys():
                        self.cur_output[arg] = self.loss_data_process(
                            arg, self.cur_data)

    def execute_model(self, training=False):
        # log.debug(dict_to_string(self.cur_data))
        self.cur_output = self.model.update(
            self.cur_data, training=training)

    def loss_data_process(self, arg: str, data: dict):
        return data[arg]

    def calc_loss_func(self, mode='train') -> None:
        self.cur_loss_debug = self.calc_loss_debug()

        if mode == 'train':
            self.cur_loss = self.calc_loss_train()
            if self.loss is None:
                self.loss = self.cur_loss['loss']
            else:
                self.loss += self.cur_loss['loss']
            self.cur_loss.update(self.cur_loss_debug)
        elif mode == 'test':
            self.cur_loss = self.cur_loss_debug

    def calc_loss_train(self) -> dict:
        if self.loss_func is None:
            raise Exception('self.loss_func is None')
        ''' some complex loss should be calculated inner model scope '''
        self.model.calc_loss(self.cur_data, self.cur_output)
        ''' standard loss can be calculated through LossFunction via cur_output '''
        ret = self.loss_func.forward(self.cur_output)
        return ret

    def calc_loss_debug(self) -> dict:
        if self.loss_func is None:
            raise Exception('self.loss_func is None')
        with torch.no_grad():
            ret = self.loss_func.forward_debug(
                self.cur_output)
        return ret

    # def gather_final_info_tensorboard(self):
    #     ret = {}
    #     config = self.config["avg_info_epoch"]['test']
    #     for scalar_name in config:
    #         info = self.get_avg_info(scalar_name)
    #         if info is None:
    #             continue
    #         else:
    #             ret[scalar_name] = info
    #     return ret

    # def gather_final_info_file(self):
    #     ret = self.gather_final_info_tensorboard()
    #     return ret

    # def gather_initial_info_tensorboard(self):
    #     ret = {}
    #     ret['batch_size'] = self.batch_size
    #     if self.model is not None:
    #         ret['infer_time'] = self.model.get_infer_time()
    #         ret['net_parameter_num'] = self.model.get_net_parameter_num()
    #     return ret

    # def gather_initial_info_file(self):
    #     ret = self.gather_initial_info_tensorboard()
    #     ret['active_loss'] = self.active_loss_funcs
    #     if self.model is not None:
    #         ret['net_struct'] = self.model.get_net()
    #     return ret

    def write_text_info(self, text_dict_to_tensorboard=None, text_dict_to_file=None,
                        log_name="info", file_line_mid="\n", step=0, enable_log=False):
        if text_dict_to_tensorboard is not None:
            tmp_lines = dict_to_string_join(
                text_dict_to_tensorboard, sep="  \n")
            assert self.writer is not None
            self.writer.add_text(log_name, tmp_lines, global_step=step)
            log.info("[{}]: add ({}) to tensorboard as tag \"{}\"".format(
                self.class_name, list(text_dict_to_tensorboard.keys()), log_name))
        if text_dict_to_file is not None:
            tmp_lines = dict_to_string_join(text_dict_to_file, mid=file_line_mid)
            file_path = "{}/{}.log".format(self.log_path, log_name)
            write_text_to_file(file_path, tmp_lines, "a")
            if enable_log:
                log.info("[{}]: add text to file: \"{}\"".format(
                    self.class_name, file_path))

    def write_tensorboard_step(self, mode="train") -> None:
        if mode == "train":
            if should_active_at(self.step, self.total_step, self.end_epoch * self.config['log']['train_image_epoch_sum'],
                                offset=1,
                                last=self.last_write_debug_info_step):
                self.write_to_tensorboard_image(mode=mode)
                self.last_write_debug_info_step = self.step
            if should_active_at(self.step, self.step_per_epoch, self.config['log']['train_scalar_epoch_sum']):
                self.write_to_tensorboard_scalar('step', mode=mode)
        elif mode == "test":
            # log.debug(self.config['log']['test_image_epoch_sum'])
            # log.debug(f"{self.step} {self.total_step}")
            if should_active_at(self.step, self.total_step, self.config['log']['test_image_epoch_sum'], offset=1):
                self.write_to_tensorboard_image(mode=mode)
            self.write_to_tensorboard_scalar('step', mode=mode, step=self.step)

    def write_to_tensorboard_scalar(self, step_mode, mode="train", step=None):
        '''
        step_mode: 'step', 'epoch',
        mode: 'train', 'test'
        '''
        assert step_mode == 'step' or step_mode == 'epoch'
        assert mode == 'train' or mode == 'test'
        if mode == 'test' and step_mode == 'epoch':
            step = 0
        if step_mode == 'step':
            if step is None:
                step = self.step
        elif step_mode == 'epoch':
            if step is None:
                step = self.epoch_index
        config = self.config[f"tensorboard_info_{step_mode}"][mode]
        written_scalar_name = []
        for scalar_name in config:
            if step_mode == 'step':
                info = self.get_info(scalar_name)
            elif step_mode == 'epoch':
                info = self.get_avg_info(scalar_name)
            if info is None:
                pass
            else:
                assert self.writer is not None
                self.writer.add_scalar(
                    "{}_{}_{}".format(step_mode, mode, scalar_name), info, global_step=step)
                written_scalar_name.append(scalar_name)
                if step_mode == 'step':
                    self.log_scalar_step_count += 1
        if step_mode == 'epoch':
            log.debug("write_epoch_{}: {} to tb".format(mode, written_scalar_name))

    def get_tensorboard_image(self, mode='train') -> dict:
        self.images = []
        self.image_texts = []
        self.debug_images = []
        self.debug_image_texts = []
        self.gather_tensorboard_image(mode=mode)
        self.gather_tensorboard_image_debug(mode=mode)
        with torch.no_grad():
            imgs = np.concatenate(self.images, 1)
            if len(self.debug_images) > 0:
                # log.debug(self.debug_image_texts)
                debug_imgs = np.concatenate(self.debug_images, 1)
            else:
                debug_imgs = None

            text = ""
            debug_text = ""
            if mode == 'train':
                text = 'train/s{}_e({}/{})'.format(self.step, self.epoch_index, self.end_epoch)
                text += ' batch:{}/{}'.format(self.batch_index, self.step_per_epoch)
            elif mode == 'test':
                text = 'test/'

            debug_text = 'debug_' + text

            scene_name = self.cur_data['metadata']['scene_name'][0]  # type: ignore
            index = self.cur_data['metadata']['index'][0]  # type: ignore
            text += f" img_{scene_name}_{index}"
            debug_text += f" img_{scene_name}_{index}"

            if mode == 'train' or mode == 'test':
                text += " ({})".format(", ".join(self.image_texts))
                debug_text += " ({})".format(", ".join(self.debug_image_texts))

        return {
            'imgs': imgs,
            'text': text,
            'debug_imgs': debug_imgs,
            'debug_text': debug_text
        }

    def gather_tensorboard_image_debug(self, mode='train'):
        pass

    def write_to_tensorboard_image(self, mode="train") -> None:
        image_data = self.get_tensorboard_image(mode=mode)
        assert self.writer is not None
        if image_data['imgs'] is not None:
            self.writer.add_image(
                image_data['text'],
                image_data['imgs'],
                self.step,
                dataformats='HWC')
            # log.debug("write_img {} {}".format(
            # self.config['job_name'], self.step))
            self.log_image_step_count += 1

        if image_data.get('debug_imgs', None) is not None:
            self.writer.add_image(
                image_data['debug_text'],
                image_data['debug_imgs'],
                self.step,
                dataformats='HWC')
            self.log_image_step_count += 1

    def log_data_step(self):
        log.debug(dict_to_string(self.cur_data, "cur_data", mmm=True))
        log.debug(dict_to_string(self.cur_output, "cur_output", mmm=True))
        log.debug(dict_to_string(self.cur_loss, "cur_loss", mmm=True))

        # self.cur_data['history_warped_scene_color_no_st_0'] = warp(
        #     self.cur_data['history_scene_color_no_st_0'], self.cur_data['merged_motion_vector_0'])

        # def write_data(data):
        #     scene_name = data['metadata']['scene_name'][0]
        #     index = data['metadata']['index'][0]
        #     write_path = f"../output/images/trainer_debug/{scene_name}_{index}/"
        #     create_dir(write_path)
        #     for k in data.keys():
        #         if not (isinstance(data[k], torch.Tensor)) or len(data[k].shape) != 4:
        #             continue
        #         if data[k].shape[1] == 1:
        #             write_buffer("{}/{}_{}.exr".format(write_path, k, index),
        #                          align_channel_buffer(
        #                 data[k][0], channel_num=3, mode="repeat"))
        #         else:
        #             write_buffer("{}/{}_{}.exr".format(write_path, k, index),
        #                          data[k][0])

        # write_data(self.cur_data)
        # self.cur_output['metadata'] = self.cur_data['metadata']
        # write_data(self.cur_output)

    def backward(self) -> None:
        def reduce_tensor(loss):
            ws = self.config['num_gpu']
            with torch.no_grad():
                torch.distributed.reduce(loss, dst=0)
            return loss / ws

        if self.scheduler is None:
            self.cur_lr = get_learning_rate(self)
            assert self.optimizer is not None
            for params_group in self.optimizer.param_groups:
                params_group['lr'] = self.cur_lr
        # self.loss = 0 # ((self.cur_output['pred'] - self.cur_output['gt']) ** 2).mean()
        if self.loss is not None and float(self.loss) < 1e-9:
            self.loss = None
            return
        if self.loss is not None:
            flag = torch.isnan(self.loss).any() or torch.isinf(self.loss).any()
            if self.enable_log and (self.debug_data_flow or flag):
                if torch.isnan(self.loss).any():
                    log.debug("=" * 20 + "loss is nan" + "=" * 20)
                    log.warn("epoch: {}, step: {}, loss is nan, skip.".format(
                        self.epoch_index, self.step))
                    self.log_data_step()
                    self.loss = None
                    exit(0)
            if float(self.loss) >= 1e-9:
                assert self.optimizer is not None
                # torch.distributed.barrier()
                # if self.config['use_ddp']:
                #     self.loss = reduce_tensor(self.loss)
                # self.optimizer.zero_grad(set_to_none=True)
                self.optimizer.zero_grad(set_to_none=True)
                # log.debug("before backward")
                self.scaler.scale(self.loss).backward()  # type: ignore
                # self.loss.backward()
                # self.optimizer.step()
                self.scaler.step(self.optimizer)  # type: ignore
                self.scaler.update()  # type: ignore
                # log.debug("after step")
                if self.scheduler is not None:
                    self.scheduler.step()
                del self.loss
            else:
                del self.loss
        self.loss = None

    def add_diff_buffer(self, name1, name2, scale=10, allow_skip=True, debug=False, cur_scale=1.0):
        buffer1 = self.get_buffer(name1, allow_skip=allow_skip)
        buffer2 = self.get_buffer(name2, allow_skip=allow_skip)
        if buffer1 is not None and buffer2 is not None:
            diff = scale * ((buffer1 - buffer2)**2)
            self.add_render_buffer(
                "{}*l2({},{})".format(scale, name1, name2), diff, debug=debug, cur_scale=cur_scale)
        else:
            assert allow_skip

    def gather_tensorboard_image(self, mode='train'):
        diff_scale = 10
        self.add_render_buffer("pred")
        self.add_render_buffer("gt")
        pred = self.get_buffer("pred", allow_skip=False, device="cuda")
        gt = self.get_buffer("gt", allow_skip=False, device="cuda")
        if pred is not None and gt is not None:
            diff = diff_scale * ((pred - gt)**2)
            self.add_render_buffer(f"diff ({diff_scale}x", buffer=diff)
            self.image_texts.insert(0, f'lpips: {float(lpips(pred, gt)):.4g}')
            self.image_texts.insert(0, f'ssim: {ssim(pred, gt):.4g}')
            self.image_texts.insert(0, f'psnr: {psnr(pred, gt):.4g}')

    def get_buffer(self, name, allow_skip=True, device="cpu") -> torch.Tensor | None:
        buffer = None
        if name in self.cur_output.keys():
            buffer = self.cur_output[name]
        if buffer is None and name in self.cur_data.keys():
            buffer = self.cur_data[name]
        if isinstance(buffer, torch.Tensor):
            if device == 'cpu':
                return torch.narrow(buffer, 0, 0, 1).detach().cpu().float()
            else:
                return torch.narrow(buffer.to(device), 0, 0, 1).detach()
        else:
            if not allow_skip:
                raise Exception(
                    f'"{name}" is not found in cur_data and cur_output\n'
                    + f'==> cur_data.keys:{list(self.cur_data.keys())}\n'
                    + f'==> cur_output.keys:{list(self.cur_output.keys())}')
            return None

    def add_render_buffer(self, name, buffer=None, buffer_type="base_color", debug=False, cur_scale=1.0):
        if buffer is None:
            buffer_cpu = self.get_buffer(name)
            if buffer_cpu is None:
                return
        elif isinstance(buffer, torch.Tensor):
            buffer_cpu = buffer.detach().cpu()
        else:
            raise Exception(f'buffer type is "{type(buffer)}", but only torch.Tensor was supported!')
        if len(buffer_cpu.shape) == 4:
            buffer_cpu = buffer_cpu[0]
        if debug:
            buffer_cpu = resize(buffer_cpu.unsqueeze(0), 0.5/cur_scale)[0]
        else:
            buffer_cpu = resize(buffer_cpu.unsqueeze(0), 1.0/cur_scale)[0]
        if buffer_cpu.shape[0] == 1:
            buffer_cpu = align_channel_buffer(buffer_cpu, channel_num=3, mode="repeat")
        buffer_torch = buffer_data_to_vis(
            align_channel_buffer(buffer_cpu), buffer_type)
        if buffer_type in ['base_color', 'scene_color', 'scene_light']:
            buffer_torch = gamma(buffer_torch)
        buffer = to_numpy(buffer_torch)
        if debug:
            # log.debug(f'name: {name}, shape: {buffer.shape}')
            self.debug_images.append(buffer)
            self.debug_image_texts.append(name)
        else:
            self.images.append(buffer)
            self.image_texts.append(name)


def get_ratio_cos(progress):
    return 0.5 * (1.0 + np.cos(progress * math.pi))


def get_learning_rate(trainer: TrainerBase) -> float:
    lr_config = trainer.train_config['lr_config']
    assert not('warp_up' in lr_config.keys() and 'warp_up_epoch' in lr_config.keys())
    if 'warm_up' in lr_config.keys():
        warm_up_scope = (
            trainer.end_epoch * lr_config['warm_up']) * trainer.step_per_epoch
    if 'warm_up_epoch' in lr_config.keys():
        warm_up_scope = (
            lr_config['warm_up_epoch']) * trainer.step_per_epoch
    else:
        warm_up_scope = 0
    if 'decay_at' in lr_config.keys():
        decay_at_scope = (trainer.end_epoch * lr_config['decay_at']) * trainer.step_per_epoch
    else:
        decay_at_scope = 0
    low = lr_config.get('low', 1e-6)
    value = lr_config['value']

    assert warm_up_scope <= decay_at_scope

    if trainer.step < warm_up_scope:
        mul = trainer.step / warm_up_scope
        low = lr_config['warm_up_low']
    elif trainer.step <= decay_at_scope:
        mul = 1
    else:
        start_step = max(decay_at_scope, warm_up_scope)
        if lr_config['mode'] == "cos":
            progress = (trainer.step - start_step) / (trainer.total_step - start_step)
            mul = get_ratio_cos(progress)
        elif lr_config['mode'] == 'flat':
            mul = 1
        else:
            raise NotImplementedError("lr_mode \"{}\" is not implemented".format(
                trainer.train_config['lr_mode']))
    return (value - low) * mul + low
