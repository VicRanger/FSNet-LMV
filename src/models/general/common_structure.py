import torch
import torch.nn as nn
import torch.nn.functional as f
from utils.utils import add_metaname
from utils.str_utils import dict_to_string
from utils.log import log
from utils.model_utils import calc_2d_dim


def create_act_func(name="relu", out_channel=None):
    ops = {
        'prelu': (lambda: nn.PReLU(out_channel)) if out_channel else (lambda: nn.PReLU()),
        'relu_': lambda: nn.ReLU(inplace=True),
        'relu': lambda: nn.ReLU(inplace=False),
        'elu': lambda: nn.ELU(),
        'sigmoid': lambda: nn.Sigmoid(),
        'softmax_2d': lambda: nn.Softmax2d()
    }
    return ops[name]


def create_norm_func(name="batch_norm_2d"):
    ops = {
        'batch_norm_2d': lambda x: torch.nn.BatchNorm2d(num_features=x),
        'instance_norm_2d': lambda x: torch.nn.InstanceNorm2d(num_features=x),
    }
    return ops[name]


class NetBase(nn.Module):
    class_name = "NetBase"
    cnt_instance = 0
    
    def __init__(self, config={}):
        add_metaname(self, NetBase)
        super().__init__()
        log.debug("init class: {}".format(self.full_name))
        self.config = config
        self.act_func_name = config.get('act_func', None)
        self.norm_func_name = config.get('norm_func', None)
        
        ''' output_type = ['inference', 'train', 'debug'] '''
        self.output_type = "debug"
    
    
    def return_output_by_type(self, ret):
        match self.output_type:
            case 'inference':
                return self.get_inference_output(ret)
            case 'train':
                return self.get_train_output(ret)
            case 'debug':
                return self.get_debug_output(ret)
            case others:
                raise Exception(f'Wrong output_type: "{others}"!')
            
    def set_eval(self):
        self.eval()
        self.set_output_type('inference')
        
    def set_train(self):
        self.train()
        self.set_output_type('train')
            
    def set_output_type(self, output_type='train') -> None:
        assert output_type in ['inference', 'train', 'debug']
        self.output_type = output_type        
        
    def get_inference_output(self, ret):
        log.debug(dict_to_string(ret, "infer"))
        return ret
    
    def get_train_output(self, ret):
        log.debug(dict_to_string(ret, "train"))
        return ret

    def get_debug_output(self, ret):
        log.debug(dict_to_string(ret, "debug"))
        return ret


class Inputable(NetBase):
    cnt_instance = 0
    class_name = "Inputable"

    def __init__(self, config={}, *args, **kwargs):
        add_metaname(self, Inputable)
        NetBase.__init__(self, config)
        self.in_channel = 0
        self.get_in_channel(config)

    def get_in_channel(self, config):
        flag = False
        if 'in_channel' in config.keys():
            self.in_channel = config['in_channel']
            log.debug("[{}] use preview module output channel: {}".format(
                self.name, self.in_channel))
            flag = True
        if 'input_buffer' in config.keys() and type(config['input_buffer']) == list and len(config['input_buffer']) > 0:
            self.in_channel += calc_2d_dim(config['input_buffer'])
            flag = True
        if 'input_buffer_' in config.keys() and type(config['input_buffer_']) == list and len(config['input_buffer_']) > 0:
            self.in_channel += calc_2d_dim(config['input_buffer_'])
            flag = True
        if not flag:
            raise SyntaxError(
                'no "in_channel"(num) or "input_buffer"(array) specified in config: {}'.format(config))


class Outputable(NetBase):
    name = "Outputable"
    cnt_instance = 0

    def __init__(self, config={}, *args, **kwargs):
        add_metaname(self, Outputable)
        NetBase.__init__(self, config)
        self.out_channel = 0
        self.out_items = config.get('output_buffer', [])
        self.get_out_channel(config)

    def get_out_channel(self, config: dict):
        if 'out_channel' in config.keys():
            self.out_channel = config['out_channel']
            log.debug("[{}] use preview module output channel: {}".format(
                self.name, self.out_channel))
        elif len(self.out_items) > 0:
            for item in self.out_items:
                self.out_channel += item['channel']
        else:
            raise SyntaxError(
                'no "out_channel"(num) or "output"(array) specified in config: {}'.format(config))

    def split_output_to_dict(self, output, prefix="", postfix=""):
        ret = {}
        cur_channel = 0
        for item in self.out_items:
            ret["{}{}{}".format(prefix+"_" if len(prefix) > 0 else "",
                item['name'],
                "_"+postfix if len(postfix) > 0 else "")] \
                = output[:, cur_channel:cur_channel+item['channel'], ...]
            cur_channel += item['channel']
        return ret
