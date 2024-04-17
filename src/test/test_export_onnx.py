from copy import deepcopy
import copy
import datetime
import time
from turtle import forward
import torch
import argparse
import os
import sys
import includes.importer
from utils.warp import warp
from utils.dataset_utils import data_to_device
from utils.config_enhancer import enhance_train_config, initialize_recipe
from dataloaders.raw_data_importer import tensor_as_type_str
from utils.str_utils import dict_to_string
from utils.parser_utils import create_json_parser, create_py_parser
from utils.log import add_prefix_to_log, log, shutdown_log
from trainers.ldr_interp_trainer import LDRInterpTrainer
from models.shade_net.shade_net_v5d4 import ShadeNetModelV5d4
from models.extra_net.extra_net_reimpl import ExtraNetModel
from models.extrass.essenet import ESSNetEModel
from models.rife.rife import RifeV6Model
from models.ifrnet.ifrnet_s import IFRNetSModel
from models.dmvfn.arch import DMVFNModel
from config.config_utils import parse_config

def convert_onnx(model, patch_loader=None):
    # model.set_eval()
    model.set_eval()
    model_input =  model.dummy_net_input
    # net = model
    net = model.get_net()
    # net.requires_grad_(False)
    model_input = data_to_device(model_input, device='cuda:0')
    model_output = model.dummy_output
    log.debug(dict_to_string(model_input))
    log.debug(dict_to_string(model_output))
    with torch.no_grad():
        torch.onnx.export(net,  # model being run
                        (model_input,{}),  # model input (or a tuple for multiple inputs)
                        "../output/model.onnx",  # where to save the model
                        export_params=True,  # store the trained parameter weights inside the model file
                        opset_version=17,  # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names=list(model_input.keys()),  # the model's input names
                        output_names=model_output.keys(),  # the model's output names
                        verbose=False,
                        )
    torch.save(model_input, "../output/model_input.pt")
    torch.save(model_output, "../output/model_output.pt")
    from onnxsim import simplify
    import onnx
    onnx_model = onnx.load("../output/model.onnx")  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, "../output/model_sim.onnx")
    
def update_config(config):
    config['use_ddp'] = config['num_gpu'] > 1
    config["use_cuda"] = config['num_gpu'] > 0
    config['device'] = "cuda:0" if config["use_cuda"] else "cpu"
    assert config['train_parameter']['batch_size'] % max(config['num_gpu'], 1) == 0
    config['train_parameter']['batch_size'] = config['train_parameter']['batch_size'] // max(config['num_gpu'], 1)
    assert config['dataset']['train_num_worker_sum'] % max(config['num_gpu'], 1) == 0
    config['dataset']['train_num_worker'] = config['dataset']['train_num_worker_sum'] // max(config['num_gpu'], 1)
    
import torch.nn as nn
class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        h, w = 540, 960
        self.dummy_input = {
            'base_color': torch.rand(1, 3, h, w).cuda().half(),
            'normal': torch.rand(1, 3, h, w).cuda().half(),
            'motion_vector': torch.rand(1, 2, h, w).cuda().half(),
            'history_scene_color_0': torch.rand(1, 3, h, w).cuda().half(),
        }
        self.prelu = nn.PReLU()
    def forward(self, data):
        b, c, h, w = data['motion_vector'].shape
        device = data['motion_vector'].device
        mvs = [data['motion_vector'] + torch.rand(b, c, h, w, device=device).half() for i in range(16)]
        warped_bcs = [warp(data['base_color'], mvs[i], padding_mode='border') for i in range(16)]
        warped_ns = [warp(data['normal'], mvs[i], padding_mode='border') for i in range(16)]
        g_sims = [torch.exp(-((warped_bcs[i].mean(dim=1, keepdim=True)+warped_ns[i].mean(dim=1, keepdim=True))**2) /2/(0.005**2)) for i in range(16)]
        pos_sims = [torch.exp(-(mvs[i][:,:1,...]**2 + mvs[i][:,1:,...]**2)) for i in range(16)]
        return g_sims[0] + pos_sims[0]
        
def BN_convert_float(module):
    """
    Utility function for network_to_half().
    Retained for legacy purposes.
    """
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine is True:
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trainer")
    parser.add_argument("--config", help="trainer config file path")
    args = parser.parse_args()
    
    config = parse_config(args.config, root_path="")
    update_config(config)
    enhance_train_config(config)
    
    config_onnx = deepcopy(config)
    config_onnx["model"]["export_onnx"] = True
    config_onnx["model"]["precision"] = "fp16"
    # config_onnx["model"]["precision"] = "fp16"
    # config_onnx["inital_inference"] = False
    model = eval(config_onnx['model']['class'])(config_onnx)
    model.set_eval()
    # model = TestNet().cuda()
    # model(model.dummy_input)
    convert_onnx(model)