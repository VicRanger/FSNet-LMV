import argparse
import torch
import numpy as np
import os
import sys
import copy
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import includes.importer
from utils.config_enhancer import enhance_train_config, update_config
from utils.flow_vis import mv_to_image
from utils.buffer_utils import aces_tonemapper, gamma
from models.loss.loss import LossFunction
from utils.dataset_utils import get_input_filter_list
from utils.utils import create_dir, remove_all_in_dir
from trainers.shade_net_trainer import ShadeNetV5Trainer
from utils.buffer_utils import gamma_log
from utils.dataset_utils import resize
from utils.dataset_utils import create_warped_buffer
from trainers.shade_net_trainer import ShadeNetV5d4Trainer
import torch.distributed as dist
from utils.warp import warp
from utils.utils import write_text_to_file
from models.shade_net.shade_net_v5d4 import ShadeNetModelV5d4
from utils.buffer_utils import align_channel_buffer
from utils.buffer_utils import write_buffer
from utils.utils import Accumulator
from utils.utils import del_data
from dataloaders.patch_loader import PatchLoader
from config.config_utils import parse_config
from utils.str_utils import dict_to_string
from utils.log import log

import cv2
import numpy as np
from matplotlib import pyplot as plt
num_he = 3
index = 0

def update_ours_history_encoder(tnr):
    block_size = tnr.config['vars']['block_size']
    for he_id in range(num_he):
        ''' start up with full rendering '''
        if index < 5:
            break
        ''' let the frames which would do the pred_recurrent under basic rule'''
        if block_size == 1:
            if he_id !=1:
                log.debug(f"@{index}.{he_id}: no pred streamed")
                continue
            else:
                log.debug(f"@{index}.{he_id}: pred being streamed")
        ''' index-he_id-1 % block_size !=0: render a frame in every ${block_size} frames '''
        assert len(tnr.last_output) > he_id
        if 'v5' in tmp_trainer.config['model']['model_name'] and tnr.model.get_net().enable_recurrent\
            or 'v6' in tmp_trainer.config['model']['model_name'] and tnr.model.get_net().enable_recurrent_d2e:
            tnr._recurrent_process_each_his_encoder(he_id)
        ''' index>block_size: warm up in first ${num_he} frames '''
        tnr._update_one_batch_recurrent_input(he_id)


def inference():
    global index
    global mode
    global writer
    log.debug(f"{'='*20} start inference {'='*20}")
    with tqdm(dataset_trainer.test_loader) as epoch_loop:
        for dataset_trainer.batch_index, part in enumerate(epoch_loop):
            for data in part:
                for tnr in trainers:
                    epoch_loop.set_description_str(f"[inference: {inference_name}_{tnr.config['model']['model_name']}_{tnr.config['vars']['block_size']}]")
                    tnr.load_data(data)
                    # log.debug(dict_to_string(tnr.cur_data))
                    update_ours_history_encoder(tnr)
                    tmp_metric = {}
                    data_skip = False
                    tnr.execute_model(training=False)
                    gt = tnr.cur_output['gt'][0]
                    # log.debug(dict_to_string(tnr.cur_output, mmm=True))

                    if tnr.config['vars']['block_size']>1 and index>num_he and index%tnr.config['vars']['block_size']==0:
                        pred = gt
                        data_skip = True
                        for item in metric:
                            tmp_metric[item] = tnr.config['vars'][f"{item}_acc"].last_data
                    else:
                        pred = tnr.cur_output['pred'][0]

                    pred = aces_tonemapper(pred)
                    gt = aces_tonemapper(gt)
                    if not data_skip:
                        for item in metric:
                            tmp_metric[item] = float(LossFunction.single_ops[item]([pred, gt]).mean().item())
                            tnr.config['vars'][f"{item}_acc"].add(tmp_metric[item])
                            
                    tnr._update_one_batch_cache("test")

                    if len(tnr.last_output) > 5:
                        del_data(tnr.last_output[0])
                        del tnr.last_output[0]
                    
                    ''' write step '''
                    step_dict = {}
                    step_dict['metadata'] = f"{tnr.cur_data['metadata']['scene_name'][0]}_{tnr.cur_data['metadata']['index'][0]}"
                    step_dict['metric'] = {}
                    for item in metric:
                        tnr.config['vars']['writer'].add_scalar(f"step_{item}", tmp_metric[item], global_step=index)
                        step_dict['metric'][item] = tmp_metric[item] if not data_skip else -1
                    step_dict['step'] = index
                    step_dict['model_name'] = tnr.config['model']['model_name']
                    write_text_to_file(tnr.config['vars']['step_log_file'], str(step_dict) + '\n', "a")
                    
                    ''' write epoch '''
                    epoch_dict = {}
                    epoch_dict['model_name'] = tnr.config['model']['model_name']
                    epoch_dict['num_steps'] = index
                    epoch_dict['num_preds'] = tnr.config['vars']["psnr_acc"].cnt
                    epoch_dict['metric'] = {}
                    for item in metric:
                        epoch_dict['metric'][item] = tnr.config['vars'][f"{item}_acc"].get()
                    write_text_to_file(tnr.config['vars']['epoch_log_file'], str(epoch_dict), "w")
                    log.debug(dict_to_string(epoch_dict))
                    
                    log.debug(float(LossFunction.single_ops["psnr"]([pred, gt]).mean().item()))
                    error_4x = 4 * torch.abs(pred - gt)
                    write_buffer(tnr.config['write_path']+f"dmdl_color/dmdl_color{str(index).zfill(4)}.exr", tnr.cur_data['dmdl_color'][0], mkdir=True)
                    write_buffer(tnr.config['write_path']+f"pred_scene_light_no_st/pred_scene_light_no_st_{str(index).zfill(4)}.exr", tnr.cur_output['pred_scene_light_no_st'][0], mkdir=True)
                    write_buffer(tnr.config['write_path']+f"pred/pred_{str(index).zfill(4)}.exr", pred, mkdir=True)
                    write_buffer(tnr.config['write_path']+f"error_4x/error_4x_{str(index).zfill(4)}.exr", error_4x, mkdir=True)
                    write_buffer(tnr.config['write_path']+f"gt/gt_{str(index).zfill(4)}.exr", gt, mkdir=True)
                    # write_buffer(tnr.config['write_path']+f"gt/gt_{str(index).zfill(4)}.png", gamma(gt), mkdir=True)

                    if 'pred_tmv' in tnr.cur_output.keys():
                        mv = tnr.cur_output['pred_tmv'][0]
                    elif 'pred_layer_0_st_tmv_0' in tnr.cur_output.keys():
                        mv = tnr.cur_output['pred_layer_0_st_tmv_0'][0]
                    mv = torch.nn.functional.sigmoid((torch.abs(mv) ** 0.5)*32*torch.sign(mv))
                    mv = mv_to_image(mv-0.5).float()/255.0
                    write_buffer(tnr.config['write_path']+f"mv_st/mv_st_{str(index).zfill(4)}.exr",
                        align_channel_buffer(mv, channel_num=3, mode="value", value=0.5), mkdir=True)
                    tnr.cur_data_index = index
            index += 1
            
def update_inference_config(config):
    config['_input_config'] = copy.deepcopy(config)
    update_config(config)
    config['local_rank'] = 0
    config['use_ddp'] = False
    config["use_cuda"] = config['num_gpu'] > 0
    config['device'] = "cuda:0" if config["use_cuda"] else "cpu"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="trainer")
    args = parser.parse_args()
    num_he = 3

    metric = ['psnr', 'ssim', 'lpips']
    dataset_cfg = parse_config("config/shadenet_v5d4_inference.yaml")
    config_path = [
        "config/shadenet_v5d4_inference.yaml"
    ]

    inference_name = "FC_TEST"
    dataset_cfg['dataset']['train_scene'] = [
        {"name":"FC_T/FC_TEST_720", "config":{"indice":[]}},
    ]
    dataset_cfg['dataset']['test_scene'] = [
        {"name":"FC_T/FC_TEST_720", "config":{"indice":[]}},
    ]
    update_inference_config(dataset_cfg)
    enhance_train_config(dataset_cfg)
    inference_name = inference_name
    dataset_trainer = eval(dataset_cfg['trainer']['class'])(
            dataset_cfg, None, resume=False)
    dataset_trainer.prepare('test')
    from torch.utils.tensorboard import SummaryWriter
    write_path = "../output/images/inference/"

    configs = []
    models = []
    trainers = []
    num_cfg = len(config_path)
    block_sizes = [1 for _ in range(num_cfg)]
    for i in range(len(config_path)):
        tmp_config = parse_config(config_path[i], root_path="")
        update_inference_config(tmp_config)
        enhance_train_config(tmp_config)
        configs.append(tmp_config)
        config_train = copy.deepcopy(tmp_config)
        config_train['dataset']['enable'] = False
        config_train['model']['input_buffer'] = dataset_cfg['model']['input_buffer']
        config_train['initial_inference'] = False
        tmp_model = eval(config_train['model']['class'])(config_train)
        models.append(tmp_model)
        resume = False
        tmp_trainer = eval(config_train['trainer']['class'])(
            config_train, tmp_model, resume=resume)
        tmp_trainer.prepare("test")
        tmp_trainer.config['vars'] = {}
        for item in metric:
            tmp_trainer.config['vars'][f"{item}_acc"] = Accumulator()
        tmp_trainer.config['vars']['block_size'] = block_sizes[i]
        tmp_trainer.config['write_path'] = write_path + f"{inference_name}/{tmp_trainer.config['model']['model_name']}_{str(block_sizes[i])}/"
        tmp_trainer.config['vars']['writer'] = SummaryWriter(log_dir=tmp_trainer.config['write_path'])
        tmp_trainer.config['vars']['step_log_file'] = f"{tmp_trainer.config['write_path']}/step.log"
        tmp_trainer.config['vars']['epoch_log_file'] = f"{tmp_trainer.config['write_path']}/epoch.log"

        create_dir(tmp_trainer.config['write_path'])
        remove_all_in_dir(tmp_trainer.config['write_path'])
        trainers.append(tmp_trainer)

    require_list = get_input_filter_list({
        'input_config': dataset_cfg,
        'input_buffer': dataset_cfg['model']['require_data']
    })
    loader = PatchLoader(
        dataset_cfg['dataset']['path'],
        job_config={'export_path': dataset_cfg['dataset']['path']},
        buffer_config=dataset_cfg['buffer_config'],
        require_list=require_list)
    
    inference()
