import copy
import time
from sympy import use
import torch
from torch import optim
from dataloaders.raw_data_importer import tensor_as_type_str
from utils.dataset_utils import data_to_device
from utils.str_utils import dict_to_string
from utils.log import get_local_rank, log
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from utils.model_utils import get_model_parm_nums, model_to_half
from utils.timer import Timer
import torch.nn as nn
from tqdm import tqdm

def module_load_by_path(module, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'No file found in "{path}"')
    net_data = torch.load(path)
    log.debug(net_data.keys())
    if "state_dict" in net_data.keys():
        module.load_state_dict(net_data['state_dict'])
    elif "net" in net_data.keys():
        module.load_state_dict(net_data['net'].state_dict())
    else:
        module.load_state_dict(net_data)
        
def get_input_by_type_str(data, t="fp32"):
    model_input = {}
    for k in data.keys():
        if isinstance(data[k], torch.Tensor):
            model_input[k] = tensor_as_type_str(
                data[k], t)
        elif isinstance(data[k], list):
            tmp_arr = data[k]
            if isinstance(tmp_arr[0], torch.Tensor):
                model_input[k] = tmp_arr
                for i in range(len(model_input[k])):
                    model_input[k][i] = tensor_as_type_str(
                        model_input[k][i], t)
        else:
            model_input[k] = data[k]
    return model_input

class ModelBase:
    def __init__(self, config: dict, use_cuda=False):
        self.trainer_config = config
        self.config = config['model']
        self.config['dataset'] = config['dataset']
        self.net = None
        self.ddp = False
        self.infer_time = -1
        self.use_cuda = use_cuda
        self.dummy_output = None
        self.model_name = self.config['model_name']
        self.precision = self.config['precision']
        self.instance_name = "{}({})".format(
            self.config['class'], self.model_name)
        log.info("[{}] model creating...".format(self.instance_name))
        if self.trainer_config['use_cuda']:
            self.use_cuda = True
        self.create_model()
        if self.precision == 'fp16':
            self.net = model_to_half(self.net)
        self.to_device()
        dummy_data = self.get_dummy_input(bs=1)
        if self.precision == 'fp16':
            dummy_data = get_input_by_type_str(dummy_data, "fp16")
        if self.use_cuda:
            dummy_data = data_to_device(dummy_data, device=self.trainer_config['device'])
        self.dummy_input = dummy_data
        self.dummy_net_input = self.calc_preprocess_input(self.dummy_input)

        if self.trainer_config.get("initial_inference", True) and get_local_rank() == 0:
            self.run_dummy_inference()
        log.info("[{}] model created.".format(self.instance_name))

    def create_model(self):
        pass

    def get_dummy_input(self, bs=1) -> dict:
        return {}
    
    def get_data_to_input(self, data):
        return data
    
    def get_augment_data(self, data):
        return data
    
    def run_dummy_inference(self):
        # log.debug(self.net)
        
        input_data = copy.deepcopy(self.dummy_net_input)
        self.dummy_output = self.inference_for_timing(input_data)
        num_run = 1000
        num_warm_run = 300
        log.info("[{}] start dummy_inference...".format(self.instance_name))
        log.info("[{}] warm_run:{}, run:{}".format(self.instance_name, num_warm_run, num_run))
        if self.trainer_config.get('debug',False):
            num_run = num_warm_run = 1
        with torch.no_grad():
            for _ in tqdm(range(num_warm_run)):
                input_data = copy.deepcopy(self.dummy_net_input)
                self.inference_for_timing(input_data)
                torch.cuda.synchronize()
            timer = Timer()
            for _ in tqdm(range(num_run)):
                input_data = copy.deepcopy(self.dummy_net_input)
                torch.cuda.synchronize()
                timer.start()
                self.inference_for_timing(input_data)
                torch.cuda.synchronize()
                timer.stop()
        self.infer_time = timer.get_avg_time()
        log.info("[{}] dummy_inference completed.".format(self.instance_name))
        log.info("estimated infer_time ({} runs): {}".format(num_run, (self.infer_time)))

    # only for time testing
    def inference_for_timing(self, data):
        self.set_eval()
        with torch.no_grad():
            output = self.forward(data)
        return output

    def inference(self, data):
        self.set_eval()
        with torch.no_grad():
            output = self.update(data, training=False)
        return output

    def update(self, data, training=True):
        if training:
            self.set_train()
        else:
            self.set_eval()
        net_input = self.calc_preprocess_input(data)
        output = self.forward(net_input)
        return output
    
    def calc_preprocess_input(self, data):
        return data
    
    def forward(self, net_input):
        assert self.net is not None
        output = self.net(net_input)
        return output
        
    def calc_net_output(self, data):
        return data

    def calc_loss(self, data, ret):
        pass

    def load_by_dict(self, state_dict):
        self.get_net().load_state_dict(state_dict)

    def load_by_path(self, path):
        module_load_by_path(self.get_net(), path)
        # with torch.no_grad():
        #     for p in self.get_net().parameters():
        #         p.data[torch.isnan(p.data)] = 0

    def save(self, path, state_dict=False):
        if state_dict:
            torch.save({'state_dict':self.get_net().state_dict()}, path)
        else:
            torch.save({'net': self.get_net(),}, path)
        # log.debug("[{}] model saved.".format(self.model_name))

    def to_device(self):
        assert self.net is not None
        if self.use_cuda:
            self.net.cuda(self.trainer_config['device'])

    def to_ddp(self, device_ids=[], output_device=0, find_unused_parameters=True):
        if self.ddp:
            raise Exception("Already a ddp model, please check code.")
        if len(device_ids) == 0:
            device_ids.append(output_device)
        # self.net = DDP(self.net, device_ids=device_ids, output_device=output_device, find_unused_parameters=True)
        self.net = DDP(self.net, device_ids=device_ids, output_device=output_device, find_unused_parameters=find_unused_parameters)
        self.ddp = True

    def get_net(self):
        assert self.net is not None
        if self.ddp:
            assert self.net.module is not None
            return self.net.module
        else:
            return self.net

    def get_infer_time(self):
        return self.infer_time

    def get_net_parameter_num(self):
        return get_model_parm_nums(self.get_net())
    
    def set_train(self):
        net = self.get_net()
        net.train()

    def set_eval(self):
        net = self.get_net()
        net.eval()


class ModelBaseEXT(ModelBase):
    
    def __init__(self, config: dict):
        ModelBase.__init__(self, config)
        
    def set_train(self):
        net = self.get_net()
        net.set_train()

    def set_eval(self):
        net = self.get_net()
        net.set_eval()