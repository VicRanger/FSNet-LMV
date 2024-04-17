import copy
import torch
import torch.nn as nn
from utils.utils import add_metaname
from models.general.common_structure import Outputable, Inputable, NetBase, create_act_func, create_norm_func

from utils.model_utils import calc_2d_dim
from utils.str_utils import dict_to_string
from utils.log import log


def parse_mid_out_channels(last_channel, channels, name, i):
    if len(channels) == 1:
        mid_channel = channels[0]
        out_channel = channels[0]
    elif len(channels) == 2:
        mid_channel = channels[0]
        out_channel = channels[1]
    elif len(channels) == 3:
        last_channel = channels[0]
        mid_channel = channels[1]
        out_channel = channels[2]

    else:
        raise RuntimeError(
            "len: {} of \"{}\" of {} is incorrect.".format(len(channels), name, i))
    return last_channel, mid_channel, out_channel

def single_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                groups=1, bias=True, norm_func_name=None, act_func_name=None, single_act_channel=True):
    if norm_func_name == 'batch_norm_2d':
        bias = False
    seq = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  stride, padding, dilation, groups, bias=bias),
    )
    if norm_func_name:
        seq.add_module("{}_1".format(norm_func_name),
                       create_norm_func(norm_func_name)(out_channels))
    if act_func_name:
        if single_act_channel:
            seq.add_module("{}_1".format(act_func_name),
                        create_act_func(act_func_name, out_channel=1)())
        else:
            seq.add_module("{}_1".format(act_func_name),
                        create_act_func(act_func_name, out_channel=out_channels)())
    return seq

class SingleConv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                groups=1, bias=True, norm_func_name=None, act_func_name=None, single_act_channel=True):
        super().__init__()
        self.conv = single_conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=True, norm_func_name=None, act_func_name=None)
        if norm_func_name is not None or act_func_name is not None:
            self.single_norm_act = nn.Sequential()
            if norm_func_name:
                self.single_norm_act.add_module("{}_1".format(norm_func_name),
                                                create_norm_func(norm_func_name)(out_channels))
            if act_func_name:
                if single_act_channel:
                    self.single_norm_act.add_module("{}_1".format(act_func_name),
                                                create_act_func(act_func_name, out_channel=1)())
                else:
                    self.single_norm_act.add_module("{}_1".format(act_func_name),
                                                create_act_func(act_func_name, out_channel=out_channels)())
    def forward(self, data, skip_add=None):
        if skip_add is not None:
            return self.single_norm_act(self.conv(data) + skip_add)
        else:
            return self.single_norm_act(self.conv(data))
        
    

def dual_conv(seq, in_channel, mid_channel, out_channel, strides=[1, 1], bias=True, act_func_name=None, norm_func_name=None, single_act_channel=True):
    if norm_func_name == 'batch_norm_2d':
        bias = False
    seq.add_module("conv1", nn.Conv2d(
        in_channel, mid_channel, kernel_size=3, stride=strides[0], padding=1, bias=bias))
    if norm_func_name is not None:
        seq.add_module("{}_1".format(norm_func_name),
                       create_norm_func(norm_func_name)(mid_channel))
    if act_func_name is not None:
        if single_act_channel:
            seq.add_module("{}_1".format(act_func_name),
                        create_act_func(act_func_name, out_channel=1)())
        else:
            seq.add_module("{}_1".format(act_func_name),
                        create_act_func(act_func_name, out_channel=mid_channel)())
    seq.add_module("conv2", nn.Conv2d(
        mid_channel, out_channel, kernel_size=3, stride=strides[1], padding=1, bias=bias))
    if norm_func_name is not None:
        seq.add_module("{}_2".format(norm_func_name),
                       create_norm_func(norm_func_name)(out_channel))
    if act_func_name is not None:
        if single_act_channel:
            seq.add_module("{}_1".format(act_func_name),
                        create_act_func(act_func_name, out_channel=1)())
        else:
            seq.add_module("{}_1".format(act_func_name),
                        create_act_func(act_func_name, out_channel=out_channel)())
    return seq


class InputBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, act_func_name, norm_func_name=None):
        super().__init__()
        self.seq = nn.Sequential()
        self.seq = dual_conv(self.seq, in_channel, mid_channel, out_channel,
                             act_func_name=act_func_name, norm_func_name=norm_func_name)

    def forward(self, input_layer):
        return self.seq(input_layer)


class EncoderBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, act_func_name, norm_func_name=None):
        super().__init__()
        self.seq = nn.Sequential()
        self.seq = dual_conv(self.seq, in_channel, mid_channel,
                             out_channel, strides=[2, 1], act_func_name=act_func_name, norm_func_name=norm_func_name)

    def forward(self, input_layer):
        return self.seq(input_layer)


class DecoderBlock(nn.Module):
    def __init__(self, in_channel, concat_channel, mid_channel, out_channel, act_func_name=None, norm_func_name=None):
        super().__init__()
        self.seq = nn.Sequential()
        self.seq = dual_conv(self.seq, in_channel + concat_channel, mid_channel,
                             mid_channel, act_func_name=act_func_name, norm_func_name=norm_func_name)
        self.deconv = nn.ConvTranspose2d(
            mid_channel, out_channel, kernel_size=2, stride=2)

    def forward(self, input_layer, skip_cat=None, skip_add=None):
        cur_input = input_layer
        cur_input = self.deconv(cur_input)
        if skip_cat is not None:
            cur_input = torch.cat([cur_input, skip_cat], dim=1)
        if skip_add is not None:
            cur_input = cur_input + skip_add
        cur_input = self.seq(cur_input)
        return cur_input


class OutputBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(OutputBlock, self).__init__()
        self.seq = nn.Sequential()
        self.seq.add_module("conv1x1", nn.Conv2d(
            c_in, c_out, kernel_size=1))

    def forward(self, data):
        ret = self.seq(data)
        return ret


class UNetEncoder(Inputable):
    '''
    {
        "class": "UNetEncoder",
        "skip-cat": True | False,
        "skip-add": True | False,
        "skip-layer": True | False,
        "struct": {
            "input": [24],
            "encoder": [
                [24], [32], [32, 48], [64]
            ],
        },
        "input": [],
        "act_func": "relu" | None,
        "norm_func": "instance_norm_2d" | None,
    }
    '''
    class_name = "UNetEncoder"
    cnt_instance = 0

    def __init__(self, config, *args, **kwargs):
        if "encoder_act_func" in config.keys():
            config['act_func'] = config['encoder_act_func']
        add_metaname(self, UNetEncoder)
        Inputable.__init__(self, config=config)
        self.output_prefix = self.config.get('output_prefix', "")
        self.last_channel = 0
        self.config['skip-layer'] = self.config.get('skip-layer', False) or self.config.get(
            'skip-cat', False) or self.config.get('skip-add', False)
        self.n_encoder = len(self.config['struct']['encoder'])
        self.encoders_out_channel = []
        self.init_input_layer()
        self.init_encoder_layers()

    def create_input_block(self, config):
        return InputBlock(config['in_channel'], config['mid_channel'], config['out_channel'],
                          config['act_func_name'], config['norm_func_name'])

    def create_encoder_block(self, config):
        return EncoderBlock(config['in_channel'], config['mid_channel'], config['out_channel'],
                            config['act_func_name'], config['norm_func_name'])

    def init_input_layer(self):
        self.last_channel = self.in_channel
        if 'input' not in self.config['struct'].keys():
            self.input_block = None
            return
        channels = self.config['struct']['input']

        self.last_channel, mid_channel, self.out_channel = parse_mid_out_channels(
            self.last_channel, channels, 'input', 0)

        self.input_block = self.create_input_block(
            {'in_channel': self.last_channel,
             'mid_channel': mid_channel,
             'out_channel': self.out_channel,
             'act_func_name': self.act_func_name,
             'norm_func_name': self.norm_func_name
             }
        )
        self.last_channel = self.out_channel

    def init_encoder_layers(self):
        self.encoders_out_channel.append(self.last_channel)
        self.encoders = nn.ModuleList()
        for i in range(self.n_encoder):
            channels = self.config['struct']['encoder'][i]
            self.last_channel, mid_channel, out_channel = parse_mid_out_channels(
                self.last_channel, channels, 'encoder', i)
            self.encoders.append(self.create_encoder_block(
                {'in_channel': self.last_channel,
                 'mid_channel': mid_channel,
                 'out_channel': out_channel,
                 'act_func_name': self.act_func_name,
                 'norm_func_name': self.norm_func_name,
                 'index': i
                 }
            ))
            self.encoders_out_channel.append(out_channel)
            self.config['encoders_out_channel'] = self.encoders_out_channel
            self.last_channel = out_channel

    def forward(self, data):
        sc_layers = []
        cur_input = data['input']
        if self.input_block is not None:
            cur_input = self.input_block(cur_input)
        if self.config['skip-layer']:
            sc_layers.append(cur_input)

        for i in range(len(self.encoders)):
            cur_input = self.encoders[i](cur_input)
            if self.config['skip-layer'] and i != len(self.encoders) - 1:
                sc_layers.append(cur_input)

        return {
            "{}".format(self.output_prefix) + 'output': cur_input,
            "{}".format(self.output_prefix) + 'sc_layers': sc_layers
        }


class UNetDecoder(Outputable):
    '''
    {
        "class": "UNetDecoder",
        "enable": True,
        "skip-cat": True,
        "skip-add": True,
        "skip-conn": True,
        "specular": True,
        "struct": {
            "concat": 0,
            "decoder": [
                [48], [32], [24], [24]
            ],
            "output": [6]
        },
        "output": [
            {"name": "pred_residual", "channel": 6},
        ],
        "act_func": "relu",
        "norm_func": "instance_norm_2d",
    }
    '''
    class_name = "UNetDecoder"
    cnt_instance = 0

    def __init__(self, config, *args, **kwargs):
        if "decoder_act_func" in config.keys():
            config['act_func'] = config['decoder_act_func']
        add_metaname(self, UNetDecoder)
        super().__init__(config=config)
        self.output_prefix = self.config.get('output_prefix', "")
        self.encoders_out_channel = self.config['encoders_out_channel']
        self.last_channel = 0
        self.n_decoder = len(self.config['struct']['decoder'])
        self.config['skip-cat'] = self.config.get('skip-cat', False)
        self.config['skip-add'] = self.config.get('skip-add', False)
        self.config['skip_conn_start'] = self.config.get('skip_conn_start', 1)
        self.config['skip_conn_offset'] = self.config.get('skip_conn_offset', 1)
        if "layer_output" in self.config['struct']:
            self.layer_output_channels = self.config['struct']['layer_output']
        else:
            self.layer_output_channels = None
        self.init_decoder_layers()
        self.init_output_layer()

    def create_decoder_block(self, config) -> nn.Module:
        return DecoderBlock(config['in_channel'], config['concat_channel'], config['mid_channel'], config['out_channel'],
                            config['act_func_name'], config['norm_func_name'])

    def create_output_block(self, config) -> nn.Module:
        return OutputBlock(config['in_channel'], config['out_channel'])

    def init_decoder_layers(self) -> None:
        self.decoders = nn.ModuleList()
        self.last_channel = self.encoders_out_channel[-1]
        self.last_channel += self.config['struct'].get("concat", 0)
        for i in range(self.n_decoder):
            channels = self.config['struct']['decoder'][i]
            self.last_channel, mid_channel, out_channel = parse_mid_out_channels(
                self.last_channel, channels, 'decoder', i)
            if i == self.n_decoder - 1 and 'output' not in self.config['struct'].keys():
                out_channel = self.out_channel
            concat_channel = 0
            if self.config['skip-cat'] and i >= self.config['skip_conn_start']:
                concat_channel = 0
                ''' TODO: concat i-th (from 1 to n) encoder to (num-i+1) decoder '''
                # if 'concat' in self.config['struct']:
                #     concat_channel += self.config['struct']['concat']
                ''' -(i+2) because ind start from 0(1st "+1" ) and last encoder is not counted (2nd "+1" )'''
                concat_channel += self.encoders_out_channel[-(i + self.config['skip_conn_offset'])]
            if self.layer_output_channels is not None and i > 0:
                layer_output_channel = self.layer_output_channels[i - 1]
            else:
                layer_output_channel = 0
            # log.debug(f"{concat_channel} {self.last_channel}")
            self.decoders.append(self.create_decoder_block(
                {'in_channel': self.last_channel - layer_output_channel,
                 'concat_channel': concat_channel,
                 'mid_channel': mid_channel,
                 'out_channel': out_channel,
                 'act_func_name': self.act_func_name,
                 'norm_func_name': self.norm_func_name,
                 'index': i
                 }
            ))
            self.last_channel = out_channel

    def init_output_layer(self) -> None:
        if 'output' not in self.config['struct'].keys():
            self.output_block = None
            return
        channels = self.config['struct']['output']
        self.config['struct']['output'][-1] = self.out_channel
        self.last_channel, mid_channel, out_channel = parse_mid_out_channels(
            self.last_channel, channels, 'output', 0)
        self.output_block = self.create_output_block(
            {'in_channel': self.last_channel,
             'mid_channel': mid_channel,
             'out_channel': out_channel,
             'act_func_name': self.act_func_name,
             'norm_func_name': self.norm_func_name,
             }
        )
        self.last_channel = out_channel

    def forward(self, data):
        cur_input = data['input']
        sc_layers = data['sc_layers']

        for i in range(len(self.decoders)):
            skip_conn = sc_layers[-(i + 1)
                                  ] if self.config['skip-cat'] else None
            skip_add = sc_layers[-(i + 1)] if self.config['skip-add'] else None
            cur_input = self.decoders[i](
                cur_input,
                skip_conn=skip_conn,
                skip_add=skip_add)

        ret = cur_input
        if self.output_block is not None:
            ret = self.output_block(ret)
        ret = self.split_output_to_dict(ret, prefix=self.output_prefix)
        return ret


class UNet(NetBase):
    '''
    {
        "class": "UNet",
        "struct": {
            "input": [32],
            "encoder": [
                [32], [48], [64], [80]
            ],
            'concat':0,
            "decoder": [
                [128, 96], [64, 48], [48, 32], [32]
            ],
            "output": [11]
        },
        "norm_func": "instance_norm_2d",
        "skip-conn": True,
        "input": [],
        "output": [
            {"name": "pred_residual", "channel": 3},
        ],
        "act_func": "relu",
        "norm_func": "instance_norm_2d",
    }
    '''
    class_name = "UNet"
    cnt_instance = 0

    def __init__(self, config, encoder_func=UNetEncoder, decoder_func=UNetDecoder):
        add_metaname(self, UNet)
        super().__init__(config=config)

        self.encoder = UNetEncoder(self.config)
        self.decoder = UNetDecoder(self.config)
        self.config["out_channel"] = self.decoder.last_channel

    def forward(self, data):
        encoder_output = self.encoder(data)
        cur_input = encoder_output['output']
        sc_layers = encoder_output['sc_layers']

        ret = self.decoder({
            'input': cur_input,
            'sc_layers': sc_layers
        })

        return ret
