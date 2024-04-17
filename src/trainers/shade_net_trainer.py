import copy
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from dataloaders.raw_data_importer import get_augmented_buffer
from utils.loss_utils import lpips, psnr, ssim
from utils.utils import del_data
import torch
from trainers.trainer_base import TrainerBase, get_learning_rate
from models.loss.loss import LossFunction
from utils.buffer_utils import aces_tonemapper, align_channel_buffer, buffer_data_to_vis, inv_gamma_log, to_numpy
from utils.str_utils import dict_to_string, dict_to_string_join
from utils.log import get_local_rank, log
from utils.warp import get_merged_motion_vector_from_last, warp

import json
import numpy as np
import torch
# from code.src.utils.utils import del_dict_item
from utils.buffer_utils import create_flip_data, inv_gamma_log, gamma_log
from utils.dataset_utils import data_to_device, merge_lr
from utils.str_utils import dict_to_string
from utils.log import log
import re
import os.path as osp
class ShadeNetTrainer(TrainerBase):
    def __init__(self, config, model, resume=False):
        super().__init__(config, model, resume)
        self.output_buffer = []
        self.loss_buffer = []

    def get_model_loss(self):
        log.debug(dict_to_string(f'get model loss using "lpips"'))
        return self.get_avg_info("lpips")
    
    def gather_tensorboard_image(self, mode='train'):
        diff_scale = 4
        self.add_render_buffer("pred", buffer_type='scene_color')
        self.add_render_buffer("gt", buffer_type='scene_color')
        pred = aces_tonemapper(self.get_buffer("pred", allow_skip=False))
        gt = aces_tonemapper(self.get_buffer("gt", allow_skip=False))
        diff = diff_scale * ((pred - gt)**2)
        self.add_render_buffer(f"diff ({diff_scale}x)", buffer=diff)
        self.add_render_buffer(f"diff_no_st ({diff_scale}x)", buffer=diff)
        if 'pred_st_color' in self.cur_output.keys():
            self.add_render_buffer("pred_scene_color_no_st", buffer_type='scene_color')
            self.add_render_buffer("scene_color_no_st", buffer_type='scene_color')
            pred_no_st = aces_tonemapper(self.get_buffer("pred_scene_color_no_st", allow_skip=False))
            gt_no_st = aces_tonemapper(self.get_buffer("scene_color_no_st", allow_skip=False))
            diff = diff_scale * ((pred - gt)**2)
            self.add_render_buffer("pred_st_color", buffer_type='scene_color')
            self.add_render_buffer("st_color", buffer_type='scene_color')
            pred_st = aces_tonemapper(self.get_buffer("pred_st_color", allow_skip=False))
            gt_st = aces_tonemapper(self.get_buffer("st_color", allow_skip=False))
            diff = diff_scale * ((pred_st - gt_st)**2)
            self.add_render_buffer(f"diff_st ({diff_scale}x)", buffer=diff)
            self.image_texts.insert(0, f'lpips_st: {float(lpips(pred_st, gt_st)):.4g}')
            self.image_texts.insert(0, f'ssim_st: {ssim(pred_st, gt_st):.4g}')
            self.image_texts.insert(0, f'psnr_st: {psnr(pred_st,gt_st):.4g}')
            self.image_texts.insert(0, f'lpips_no_st: {float(lpips(pred_no_st, gt_no_st)):.4g}')
            self.image_texts.insert(0, f'ssim_no_st: {ssim(pred_no_st, gt_no_st):.4g}')
            self.image_texts.insert(0, f'psnr_no_st: {psnr(pred_no_st, gt_no_st):.4g}')
        self.image_texts.insert(0, f'lpips: {float(lpips(pred, gt)):.4g}')
        self.image_texts.insert(0, f'ssim: {ssim(pred, gt):.4g}')
        self.image_texts.insert(0, f'psnr: {psnr(pred, gt):.4g}')

    def calc_loss_train(self):
        return super().calc_loss_train()

    def gather_tensorboard_image_debug(self, mode='train') -> None:
        with torch.no_grad():
            self.images = []
            self.image_texts = []

            self.debug_images = []
            self.debug_image_texts = []

            if 'mask_kth' in self.cur_output.keys():
                self.debug_images.append(
                    to_numpy(self.cur_output['mask_kth'][0]))
                self.debug_image_texts.append("mask_kth")


class ShadeNetV5Trainer(ShadeNetTrainer):
    def __init__(self, args, model, resume=False):
        super().__init__(args, model, resume)

    def gather_tensorboard_image_debug(self, mode='train') -> None:
        with torch.no_grad():
            num_he = int(self.model.get_net().num_history_encoder) # type: ignore
            num_dec = int(self.model.get_net().num_shade_decoder_layer) # type: ignore

            net = self.model.get_net()
            if net.enable_demodulate:
                albedo = self.get_buffer('dmdl_color', allow_skip=False)
                self.add_render_buffer(f"dmdl_color({self.config['dataset']['demodulation_mode']})", debug=True)
            else:
                albedo = None
                
            if self.model.get_net().method == "residual":
                self.add_render_buffer("pred_scene_color_no_st", debug=True)
                self.add_render_buffer("scene_color_no_st", debug=True)
                residual_item = self.get_buffer("residual_item", allow_skip=False)
                if residual_item is not None and albedo is not None and self.model.get_net().enable_demodulate:
                    residual_item = residual_item * albedo
                self.add_render_buffer("residual_item", buffer=residual_item, debug=True)
                if residual_item is not None:
                    self.add_render_buffer("abs(residual)", buffer=torch.abs(
                        residual_item - self.get_buffer("pred_scene_color_no_st", allow_skip=False)), debug=True)
                self.add_render_buffer("scene_color", debug=True)
                self.add_diff_buffer("gt_comp", "gt", debug=True)
                if self.model.get_net().enable_output_warp_upscale:
                    self.add_render_buffer('pred_temporal_mv_upscale', buffer_type="motion_vector_8", debug=True)
                    self.add_render_buffer('pred_tmv_upscale', buffer_type="motion_vector_8", debug=True)
                if self.model.get_net().enable_output_warp2:
                    self.add_render_buffer('pred_temporal_mv', buffer_type="motion_vector_8", debug=True)
                    self.add_render_buffer('pred_tmv', buffer_type="motion_vector_8", debug=True)
                if self.model.get_net().enable_output_warp1:
                    self.add_render_buffer(f'pred_layer_{4}_temporal_mv_{0}', buffer_type="motion_vector_8", debug=True)
                    self.add_render_buffer(f'pred_layer_{4}_tmv_{0}', buffer_type="motion_vector_8", debug=True)

            elif self.model.get_net().method == "shade":
                ''' no longer use'''
                pass

            if self.model.get_net().enable_st:
                self.add_render_buffer("pred_st_color", debug=True)
                self.add_render_buffer("st_color", debug=True)
                self.add_render_buffer("pred_st_alpha", debug=True)
                self.add_render_buffer("st_alpha", debug=True)
                self.add_render_buffer("pred_sky_color", debug=True)
                self.add_render_buffer("sky_color", debug=True)
                self.add_render_buffer("skybox_mask", debug=True)
                self.add_render_buffer("pred_comp_color_before_sky_st", debug=True)
                self.add_render_buffer("pred_comp_color_sky", debug=True)

            for he_id in range(num_he):
                # self.add_diff_buffer(f'history_warped_gt_{self.model.get_net().history_id[he_id]}',
                #         'gt', allow_skip=False, debug=True)
                if f'pred_layer_{1}_temporal_mv_{0}' in self.cur_output.keys():
                    break

                for i in range(num_dec, 0, -1):
                    if i == num_dec:
                        if self.model.get_net().enable_output_concat and \
                                self.model.get_net().shade_decoder.output_block is not None: # type: ignore
                            mv = self.cur_output[f'pred_layer_{i}_spatial_mv_{he_id}'][0]
                        else:
                            mv = None
                    else:
                        ratio = 2 ** (num_dec - i)
                        mv = F.upsample(self.cur_output[f'pred_layer_{i}_spatial_mv_{he_id}'][:1], scale_factor=ratio)[0]
                    if mv is not None:
                        self.add_render_buffer(f'pred_layer_{i}_spatial_mv_{he_id}',
                                               buffer_type="motion_vector_64", buffer=mv, debug=True)

                if self.model.get_net().enable_temporal_warped_feature:
                    for i in range(num_dec, 0, -1):
                        if i == num_dec:
                            if self.model.get_net().enable_output_concat and \
                                    self.model.get_net().shade_decoder.output_block is not None: # type: ignore
                                mv = self.cur_output[f'pred_layer_{i}_temporal_mv_{he_id}'][0]
                            else:
                                mv = None
                        else:
                            ratio = 2 ** (num_dec - i)
                            mv = F.upsample(self.cur_output[f'pred_layer_{i}_temporal_mv_{he_id}'][:1], scale_factor=ratio)[0]
                        if mv is not None:
                            self.add_render_buffer(f'pred_layer_{i}_temporal_mv_{he_id}',
                                                   buffer_type="motion_vector_8", buffer=mv, debug=True)


class ShadeNetV5d4Trainer(ShadeNetTrainer):
    def __init__(self, config, model, resume=False):
        super().__init__(config, model, resume)
        self.output_cache = None
        self.last_output = []
        self.cur_data_index = -1
        self.num_he = self.config['model']['history_encoders']['num']
        

    def gather_tensorboard_image_debug(self, mode='train') -> None:
        with torch.no_grad():
            num_he = int(self.model.get_net().num_history_encoder) # type: ignore
            num_dec = int(self.model.get_net().num_shade_decoder_layer) # type: ignore

            net = self.model.get_net()
            self.add_render_buffer("pred", buffer_type="scene_color", debug=True)
            self.add_render_buffer("scene_color", buffer_type="scene_color", debug=True)

            if net.enable_demodulate:
                albedo = self.get_buffer('dmdl_color', allow_skip=False)
                self.add_render_buffer(f"dmdl_color({self.config['dataset']['demodulation_mode']})", albedo, debug=True)
            else:
                albedo = None

            if self.model.get_net().method in ["residual", "shade"]:
                self.add_render_buffer("pred_scene_light_no_st", debug=True)
                self.add_render_buffer("scene_light_no_st", debug=True)
                residual_item = self.get_buffer("residual_item", allow_skip=False)
                self.add_render_buffer("abs(residual)", buffer=torch.abs(-(residual_item) + # type: ignore
                                                                         self.get_buffer("pred_scene_light_no_st", allow_skip=False)), 
                                       buffer_type="depth", debug=True)
                if residual_item is not None and albedo is not None and self.model.get_net().enable_demodulate:
                    residual_item = residual_item * albedo
                self.add_render_buffer("residual_item", buffer=residual_item, buffer_type="scene_color", debug=True)
                self.add_diff_buffer("gt_comp", "gt", debug=True)
                if self.model.get_net().enable_output_warp1_upscale:
                    self.add_render_buffer(f'pred_layer_{0}_tmv_{0}_upscale', buffer_type="motion_vector_8", debug=True, cur_scale=2.0)
                if self.model.get_net().enable_output_warp2:
                    self.add_render_buffer('pred_tmv', buffer_type="motion_vector_8", debug=True)
                if self.model.get_net().enable_output_warp1:
                    self.add_render_buffer(f'pred_layer_{0}_tmv_{0}', buffer_type="motion_vector_8", debug=True)

            if self.model.get_net().enable_st:
                self.add_render_buffer("pred_st_color", buffer_type="scene_color", debug=True)
                self.add_render_buffer("st_color", buffer_type="scene_color", debug=True)
                self.add_render_buffer("pred_st_alpha", debug=True)
                self.add_render_buffer("st_alpha", debug=True)
                self.add_render_buffer("pred_sky_color", buffer_type="scene_color", debug=True)
                self.add_render_buffer("sky_color", buffer_type="scene_color", debug=True)
                self.add_render_buffer("skybox_mask", debug=True)
                self.add_render_buffer("pred_comp_color_before_sky_st", buffer_type="scene_color", debug=True)
                self.add_render_buffer("pred_comp_color_sky", buffer_type="scene_color", debug=True)

            def get_pyramid_buffer(layer_id, he_id, in_smv_name):
                if f'pred_layer_{layer_id}_{in_smv_name}_{he_id}' not in self.cur_output.keys():
                    return None
                if i == num_dec:
                    mv = self.cur_output[f'pred_layer_{layer_id}_{in_smv_name}_{he_id}'][0]
                else:
                    ratio = 2 ** (layer_id)
                    mv = F.upsample(self.cur_output[f'pred_layer_{layer_id}_{in_smv_name}_{he_id}'][:1], scale_factor=ratio)[0]
                return mv

            for he_id in range(num_he):
                if self.model.get_net().enable_smv0:
                    for i in range(num_dec):
                        self.add_render_buffer(f'l{i}_smv0_{he_id}', buffer_type="motion_vector_64",
                                               buffer=get_pyramid_buffer(i, he_id, "smv0"), debug=True)
                        if he_id == 0:
                            if self.model.get_net().enable_sep_smv_res:
                                self.add_render_buffer(f'l{i}_smv_res_{he_id}', buffer_type="motion_vector_64",
                                                       buffer=get_pyramid_buffer(i, he_id, "smv_res"), debug=True)

                if self.model.get_net().enable_temporal_warped_feature:
                    for i in range(num_dec):
                        self.add_render_buffer(f'l{i}_tmv_{he_id}', buffer_type="motion_vector_64",
                                               buffer=get_pyramid_buffer(i, he_id, "tmv"), debug=True)

    
    def _use_cached_history_encoding(self, he_id):
        assert he_id > 0
        self.cur_data[f"{self.model.get_net().he_pfs[he_id]}sc_layers"] = [item.detach() for item in self.last_output[-1][f'{self.model.get_net().he_pfs[he_id-1]}sc_layers']]
        self.cur_data[f"{self.model.get_net().he_pfs[he_id]}output"] = self.last_output[-1][f'{self.model.get_net().he_pfs[he_id-1]}output'].detach()
        
    def _recurrent_gbuffer_layer(self, he_id):
        self.cur_data[f"history_{he_id}_se_sc_layers"] = [item.detach().clone() for item in self.last_output[-(he_id) - 1]['se_sc_layers']]
        # self.cur_data[f"history_{he_id}_se_sc_layers"] = self.last_output[-(he_id) - 1]['se_sc_layers']

    def _recurrent_hidden_layer(self, he_id, pf=""):
        self.cur_data[f'hidden_{he_id}_{pf}sc_layers'] = [item.detach().clone() for item in self.last_output[-(he_id) - 1]['sd_sc_layers'][::-1]]
        # self.cur_data[f'hidden_{he_id}_{pf}sc_layers'] = self.last_output[-(he_id) - 1]['sd_sc_layers'][::-1]

    def _recurrent_process_each_his_encoder(self, he_id):
        self._recurrent_gbuffer_layer(he_id)
        self._recurrent_hidden_layer(he_id)
        self.cur_data[f'recurrent_feature_{he_id}'] = True
        
    def flip_data(self, data):
        def get_flip_argument():
            vertical = False
            horizontal = False
            if np.random.random() > 0.5:
                vertical = True
            if np.random.random() > 0.5:
                horizontal = True
            return vertical, horizontal
        
        def flip_(data, batch_size, flip_datas):
            for batch_id in range(batch_size):
                data = create_flip_data(data, vertical=flip_datas[batch_id][0], horizontal=flip_datas[batch_id][1], use_batch=True, batch_mask=[batch_id])
                if 'history_data_list' in data.keys():
                    history_datas = data['history_data_list']
                    for he_id, he_data in enumerate(history_datas):
                        history_datas[he_id] = create_flip_data(he_data, vertical=flip_datas[batch_id][0], horizontal=flip_datas[batch_id][1], use_batch=True, batch_mask=[batch_id])
            return data
        
        if self.cur_data_index == 0:
            self.last_flip_datas = [get_flip_argument() for _ in range(self.train_dataset.batch_size)]
            
        data = flip_(data, self.train_dataset.batch_size, self.last_flip_datas)
        data['metadata']['vertical_flip'] = torch.tensor([item[0] for item in self.last_flip_datas], device=data['metadata']['index'].device)
        data['metadata']['horizontal_flip'] = torch.tensor([item[1] for item in self.last_flip_datas], device=data['metadata']['index'].device)
        return data
    
    def load_data(self, data, mode="test"):
        if self.use_cuda:
            self.cur_data: dict = data_to_device(data, self.config['device'], non_blocking=True)  # type: ignore
        if mode == 'train' and self.config['dataset']['flip'] and self.config['dataset']['is_block_part']: 
            assert 'vertical_flip' not in self.cur_data['metadata'].keys() # type: ignore
            self.cur_data = self.flip_data(self.cur_data)
        if not self.config['dataset']['augment_loader']:
            self.cur_data = self.model.get_augment_data(self.cur_data)
        self.cur_data['cur_data_index'] = self.cur_data_index
        
    def _update_one_batch(self, epoch_index=None, batch_index=None, mode="train"):
        num_he = self.config['model']['history_encoders']['num']
        full_rendered = True
        recurrent_pred = (self.config['trainer'][f'recurrent_{mode}'])
        ''' create feature_0 and encoding_0 '''
        if mode == "train":
            probs = self.config['model']['train_probs']
        elif mode == "test":
            probs = self.config['model']['test_probs']
        else:
            raise Exception('unsupported mode: "{mode}"')
        # random select
        for he_id in range(num_he):
            he_pf = self.model.get_net().he_pfs[he_id] # type: ignore
            if self.cur_data_index <= he_id:
                continue
            # if he_id > 0 and mode == 'train':
            #     self._use_cached_history_encoding(he_id)
            self.cur_data[he_pf + "prob"] = 0 # type: ignore
            if np.random.random() > probs[he_id]:
                continue
            if self.model.get_net().enable_recurrent:
                self._recurrent_process_each_his_encoder(he_id)
            # self.cur_data[he_pf + "feature_prob"] = 1 # type: ignore
            self.cur_data[he_pf + "prob"] = 1 # type: ignore
            # self._update_one_batch_recurrent_lmv()
            if recurrent_pred:
                start_recurrent_epoch = int(self.end_epoch * self.config['trainer']['recurrent_train_start'])
                if epoch_index == start_recurrent_epoch and batch_index == 0:
                    self.min_loss = 1e9
                if not (mode == "train" and self.epoch_index < start_recurrent_epoch):
                    self._update_one_batch_recurrent_input(he_id)
                    full_rendered = False

        self.cur_data['rendered_prob'] = 1 if full_rendered else 0 # type: ignore
        self.update_forward(epoch_index=epoch_index,
                            batch_index=batch_index, mode=mode)
        if mode == "train":
            self.gather_execute_result(training=True, enable_loss=True)
        if mode == "test":
            self.gather_execute_result(enable_loss=True)

        self.update_backward(epoch_index=epoch_index,
                             batch_index=batch_index, mode=mode)
        self._update_one_batch_cache(mode)
        
    def _update_one_batch_recurrent_input(self, he_id):
        for buffer_name in self.config['model']['scene_color_encoder']['input_buffer']:
            self.cur_data[f'history_{buffer_name}_{he_id}'] = self.last_output[-1-he_id][f'pred_{buffer_name}'].detach().clone() # type: ignore
        self.cur_data[f'history_scene_color_no_st_{he_id}'] = self.last_output[-1-he_id]['pred_scene_color_no_st'].detach().clone() # type: ignore
        self.cur_data[f'recurrent_pred_{he_id}'] = True
        if f"{self.model.get_net().he_pfs[he_id]}sc_layers" in self.cur_data.keys():
            del self.cur_data[f"{self.model.get_net().he_pfs[he_id]}sc_layers"]
            del self.cur_data[f"{self.model.get_net().he_pfs[he_id]}output"]

    def _update_one_batch_cache(self, mode):
        num_he = self.config['model']['history_encoders']['num']
        recurrent_pred = (self.config['trainer'][f'recurrent_{mode}'])
        if len(self.last_output) > num_he:
            del self.last_output[0]
        cache_output = {}
        if recurrent_pred:
            cache_output.update({'pred_'+k: self.cur_output['pred_'+k].detach() for k in self.config['model']['scene_color_encoder']['input_buffer']})
            cache_output['pred_scene_color_no_st'] = self.cur_output['pred_scene_color_no_st'].detach()
        for he_id in range(num_he-1):
            cache_output[f'{self.model.get_net().he_pfs[he_id]}sc_layers'] = [item.detach() for item in self.cur_output[f'{self.model.get_net().he_pfs[he_id]}sc_layers']]
            cache_output[f'{self.model.get_net().he_pfs[he_id]}output'] = self.cur_output[f'{self.model.get_net().he_pfs[he_id]}output'].detach()
        if self.model.get_net().enable_recurrent:
            cache_output['sd_sc_layers'] = [item.detach() for item in self.cur_output['sd_sc_layers']]
            cache_output['se_sc_layers'] = [item.detach() for item in self.cur_output['se_sc_layers']]
        cache_output['metadata'] = copy.deepcopy(self.cur_data['metadata']) # type: ignore
        self.last_output.append(cache_output)
        
    def update(self, datas, epoch_index=None, batch_index=None, mode="train"):
        ''' check if its same block with last_data '''
        for i, data in enumerate(datas):
            # if get_local_rank() == 0:
            #     if len(self.last_output) > 0:
            #         log.debug("{} {}".format(self.last_output[-1]['metadata'], data['metadata']))
            is_same_block = len(
                self.last_output) > 0 and self.last_output[-1]['metadata']['scene_name'][0] == data['metadata']['scene_name'][0] \
                                    and self.last_output[-1]['metadata']['index'][0] == data['metadata']['index'][0] - 1
            if is_same_block:
                ''' continue the current block '''
                self.cur_data_index += 1
                # if get_local_rank() == 0:
                #     log.debug(f"continue with {data['metadata']['index'][0]}, index: {self.cur_data_index}.")
            else:
                ''' start a new block '''
                self.cur_data_index = 0
                del self.last_output
                self.last_output = []
                # if get_local_rank() == 0:
                #     log.debug(f"new block started. index: {self.cur_data_index}, with {data['metadata']['index'][0]}")
            self.load_data(data, mode)
            self._update_one_batch(epoch_index=epoch_index, batch_index=batch_index, mode=mode)


class ShadeNetV5d4SeqTrainer(ShadeNetV5d4Trainer):
    def __init__(self, args, model, resume=False):
        super().__init__(args, model, resume)
        self.output_cache = None
        self.last_output = []

    def update(self, data, epoch_index=None, batch_index=None, mode="train"):
        self.last_output = []
        for i, item in enumerate(data):
            if mode == 'train':
                self.cur_data_index = i
            else:
                self.cur_data_index += 1
            self.load_data(item, mode)
            self._update_one_batch(epoch_index=self.epoch_index, batch_index=self.batch_index, mode=mode)
        self.last_output.clear()


class ShadeNetV6Trainer(ShadeNetV5d4Trainer):
    def gather_tensorboard_image_debug(self, mode='train') -> None:
        with torch.no_grad():
            num_he = int(self.model.get_net().num_history_encoder) # type: ignore
            num_dec = int(self.model.get_net().num_shade_decoder_layer) # type: ignore

            net = self.model.get_net()
            self.add_render_buffer("pred", buffer_type="scene_color", debug=True)
            self.add_render_buffer("scene_color", buffer_type="scene_color", debug=True)

            if net.enable_demodulate:
                albedo = self.get_buffer('dmdl_color', allow_skip=False)
                self.add_render_buffer(f"dmdl_color({self.config['dataset']['demodulation_mode']})", albedo, debug=True)
            else:
                albedo = None

            if self.model.get_net().method in ["residual", "shade"]:
                self.add_render_buffer("pred_scene_light_no_st", debug=True)
                self.add_render_buffer("scene_light_no_st", debug=True)
                self.add_render_buffer("contin_mask", buffer_type="depth", debug=True)
                self.add_render_buffer("residual_mask", buffer_type="depth", debug=True)
                residual_item = self.get_buffer("residual_item", allow_skip=False)
                self.add_render_buffer("abs(residual)", buffer=torch.abs(-(residual_item) + # type: ignore
                                                                         self.get_buffer("pred_scene_light_no_st", allow_skip=False)), 
                                       buffer_type="depth", debug=True)
                if residual_item is not None and albedo is not None and self.model.get_net().enable_demodulate:
                    residual_item = residual_item * albedo
                self.add_render_buffer("residual_item", buffer=residual_item, buffer_type="scene_color", debug=True)
                self.add_diff_buffer("gt_comp", "gt", debug=True)
                if self.model.get_net().enable_output_block_warp:
                    self.add_render_buffer('pred_tmv', buffer_type="motion_vector_8", debug=True)
                self.add_render_buffer(f'pred_layer_{0}_tmv_{0}', buffer_type="motion_vector_8", debug=True)
                if self.model.get_net().enable_output_block_st_warp:
                    self.add_render_buffer('pred_st_tmv', buffer_type="motion_vector_8", debug=True)
                if self.model.get_net().enable_st_lmv:
                    self.add_render_buffer(f'pred_layer_{0}_st_tmv_{0}', buffer_type="motion_vector_8", debug=True)

            if self.model.get_net().enable_st:
                self.add_render_buffer("pred_st_color", buffer_type="scene_color", debug=True)
                self.add_render_buffer("st_color", buffer_type="scene_color", debug=True)
                self.add_render_buffer("pred_st_alpha", debug=True)
                self.add_render_buffer("st_alpha", debug=True)
                self.add_render_buffer("pred_sky_color", buffer_type="scene_color", debug=True)
                self.add_render_buffer("sky_color", buffer_type="scene_color", debug=True)
                self.add_render_buffer("skybox_mask", debug=True)
                self.add_render_buffer("pred_comp_color_before_sky_st", buffer_type="scene_color", debug=True)
                self.add_render_buffer("pred_comp_color_sky", buffer_type="scene_color", debug=True)

            def get_pyramid_buffer(layer_id, he_id, in_name):
                if f'pred_layer_{layer_id}_{in_name}_{he_id}' not in self.cur_output.keys():
                    return None
                if i == num_dec:
                    mv = self.cur_output[f'pred_layer_{layer_id}_{in_name}_{he_id}'][0]
                else:
                    ratio = 2 ** (layer_id)
                    mv = F.upsample(self.cur_output[f'pred_layer_{layer_id}_{in_name}_{he_id}'][:1], scale_factor=ratio)[0]
                return mv

            for he_id in range(num_he):
                if self.model.get_net().enable_lmv:
                    for i in range(num_dec):
                        self.add_render_buffer(f'l{i}_lmv_{he_id}', buffer_type="motion_vector_64",
                                               buffer=get_pyramid_buffer(i, he_id, "lmv"), debug=True)
                        if self.model.get_net().enable_st_lmv:
                            self.add_render_buffer(f'l{i}_st_lmv_{he_id}', buffer_type="motion_vector_64",
                                                buffer=get_pyramid_buffer(i, he_id, "st_lmv"), debug=True)
                        if he_id == 0:
                            if self.model.get_net().enable_lmv_res:
                                self.add_render_buffer(f'l{i}_lmv_res_{he_id}', buffer_type="motion_vector_64",
                                                       buffer=get_pyramid_buffer(i, he_id, "lmv_res"), debug=True)
                            if self.model.get_net().enable_st_lmv_res:
                                self.add_render_buffer(f'l{i}_st_lmv_res_{he_id}', buffer_type="motion_vector_64",
                                                       buffer=get_pyramid_buffer(i, he_id, "st_lmv_res"), debug=True)

                if self.model.get_net().enable_temporal_warped_feature:
                    for i in range(num_dec):
                        self.add_render_buffer(f'l{i}_tmv_{he_id}', buffer_type="motion_vector_64",
                                               buffer=get_pyramid_buffer(i, he_id, "tmv"), debug=True)
                        if self.model.get_net().enable_st_lmv:
                            self.add_render_buffer(f'l{i}_st_tmv_{he_id}', buffer_type="motion_vector_64",
                                                buffer=get_pyramid_buffer(i, he_id, "st_tmv"), debug=True)
            
    def _recurrent_decoder_layer_d2d(self):
        num_he = self.config['model']['history_encoders']['num']
        he_id = -1
        for i in range(num_he):
            if 'd2d_sc_layers' in self.last_output[-(i) - 1]:
                self.cur_data[f'he_{i}_d2d_sc_layers'] = self.last_output[-(i) - 1]['d2d_sc_layers']
                he_id = i
                break
        ''' TODO: generate fake zero feature for d2d_sc_layers '''
        if he_id == -1:
            ...
            # self.cur_data[f'he_{i}_d2d_sc_layers'] = self.last_output[-(i) - 1]['d2d_sc_layers']
                        
    def _recurrent_gbuffer_layer_d2e(self, he_id):
        self.cur_data[f"history_{he_id}_ge_sc_layers"] = [item.detach().clone() # type: ignore
                                                            for item in self.last_output[-(he_id) - 1]['ge_sc_layers']]

    def _recurrent_hidden_layer_d2e(self, he_id, pf=""):
        # ratio = 2 ** self.model.get_net().num_shade_decoder_layer # type: ignore
        # tmp_shape = self.cur_data['scene_color'].shape # type: ignore
        self.cur_data[f'hidden_{he_id}_{pf}sc_layers'] = [item.detach().clone() # type: ignore
                                                        for item in self.last_output[-(he_id) - 1]['d2e_sc_layers'][::-1]]
                        
    def _recurrent_hidden_layer_e2e(self):
        # log.debug('history_e2e_sc_layers recurrency at trainer')
        self.cur_data[f"history_e2e_sc_layers"] = [item.detach() # type: ignore
                                                            for item in self.last_output[-1][f'e2e_sc_layers']]
        # ''' TODO: use encoder to generate '''
        # h_channel = self.model.get_net().recurrent_blocks[-1].in_channel # type: ignore
        # if self.output_cache is None:
        #     self.output_cache = torch.zeros([tmp_shape[0], h_channel, int(tmp_shape[2] / ratio), int(tmp_shape[3] / ratio)])
        # self.recurrent_data[f'hidden_{he_id}_{pf}output'] = self.output_cache # type: ignore
    
    def _recurrent_process_each_his_encoder(self, he_id):
        self._recurrent_gbuffer_layer_d2e(he_id)
        self._recurrent_hidden_layer_d2e(he_id)
        self.cur_data[f'recurrent_d2e_feature_{he_id}'] = True
        if self.model.get_net().enable_st_encoder:
            self._recurrent_hidden_layer_d2e(he_id, pf="st_")
            self.cur_data[f'recurrent_d2e_st_feature_{he_id}'] = True
            

    def _update_one_batch(self, epoch_index=None, batch_index=None, mode="train"):
        num_he = self.config['model']['history_encoders']['num']
        full_rendered = True
        recurrent_pred = (self.config['trainer'][f'recurrent_{mode}'])
        ''' create feature_0 and encoding_0 '''
        if mode == "train":
            probs = self.config['model']['train_probs']
        elif mode == "test":
            probs = self.config['model']['test_probs']
        else:
            raise Exception('unsupported mode: "{mode}"')
        # random select
        recurrent_d2d_he_id = -1
        for he_id in range(num_he):
            he_pf = self.model.get_net().he_pfs[he_id] # type: ignore
            self.cur_data[he_pf + "prob"] = 0 # type: ignore
            if self.cur_data_index <= he_id:
                continue
            # if he_id > 0 and mode == 'train':
            #     self._use_cached_history_encoding(he_id)
            decoder_recurrent = np.random.random() <= probs[he_id]
            ''' recurrent_e2e '''
            if self.model.get_net().enable_recurrent_e2e and he_id == 0:
                self._recurrent_hidden_layer_e2e()
            if not decoder_recurrent:
                continue
            ''' recurrent_d2d '''
            if self.model.get_net().enable_recurrent_d2d and recurrent_d2d_he_id == -1:
                self.cur_data[f'{he_pf}d2d_sc_layers'] = [item.detach().clone() for item in self.last_output[-1-he_id]['d2d_sc_layers']]
                recurrent_d2d_he_id = he_id
                self.cur_data['recurrent_d2d_he_id'] = he_id
            ''' recurrent_d2e '''
            if self.model.get_net().enable_recurrent_d2e:
                self._recurrent_process_each_his_encoder(he_id)
            # and self.model.get_net().enable_recurrent \
            if recurrent_pred:
                start_recurrent_epoch = int(self.end_epoch * self.config['trainer']['recurrent_train_start'])
                if epoch_index == start_recurrent_epoch and batch_index == 0:
                    self.min_loss = 1e9
                if not (mode == "train" and self.epoch_index < start_recurrent_epoch):
                    self._update_one_batch_recurrent_input(he_id)
                    self.cur_data[he_pf + "prob"] = 1 # type: ignore
                    full_rendered = False
        if self.net.enable_recurrent_lmv:
            self._update_one_batch_recurrent_lmv()
        
        self.cur_data['rendered_prob'] = 1 if full_rendered else 0 # type: ignore
        self.update_forward(epoch_index=epoch_index,
                            batch_index=batch_index, mode=mode)
        if mode == "train":
            self.gather_execute_result(training=True, enable_loss=True)
        if mode == "test":
            self.gather_execute_result(enable_loss=True)

        self.update_backward(epoch_index=epoch_index,
                             batch_index=batch_index, mode=mode)

        self._update_one_batch_cache(mode)

    
    def _update_one_batch_recurrent_lmv(self):
        def update_recurrent_lmv(lmv_pf=""):
            history_num = min(len(self.last_output), self.num_he)
            tmp_data = {'motion_vector': self.cur_data['merged_motion_vector_0']}
            last_data = []
            for he_id in range(history_num):
                if self.cur_data_index <= he_id:
                    break
                he_pf = self.net.he_pfs[he_id] # type: ignore
                if self.cur_data[he_pf + "prob"] == 1:
                    last_data.append({'motion_vector': self.last_output[-1-he_id][f'pred_recurrent_{lmv_pf}tmv'].detach().clone()})
                else:
                    last_data.append({'motion_vector': self.last_output[-1-he_id][f'merged_motion_vector_0'].detach().clone()})
            # log.debug(dict_to_string([self.cur_data_index, tmp_data, last_data], "before_update"))
            get_augmented_buffer(['merged_motion_vector'], self.config['buffer_config'], tmp_data, last_data=last_data, allow_skip=False, with_batch=True, history_data_check=False)
            # log.debug(dict_to_string([tmp_data], "after_update"))
            for he_id in range(1, history_num):
                he_pf = self.net.he_pfs[he_id] # type: ignore
                if self.cur_data[he_pf + "prob"] == 1:
                    # log.debug(dict_to_string([self.cur_data_index, he_id, self.cur_data[he_pf + 'prob']]))
                    # log.debug(f"update {lmv_pf}merged_motion_vector_{he_id}")
                    self.cur_data[f'{lmv_pf}merged_motion_vector_{he_id}'] = tmp_data[f'merged_motion_vector_{he_id}']
        if self.cur_data_index == 0:
            return
        update_recurrent_lmv()
        if self.net.enable_st_lmv or self.net.enable_st_lmv_res:
            update_recurrent_lmv("st_")

    def _update_one_batch_cache(self, mode):
        num_he = self.config['model']['history_encoders']['num']
        recurrent_pred = (self.config['trainer'][f'recurrent_{mode}'])
        if len(self.last_output) > num_he:
            del self.last_output[0]
        cache_output = {}
        net = self.model.get_net()
        if recurrent_pred:
            cache_output.update({'pred_'+k: self.cur_output['pred_'+k].detach() for k in self.config['model']['scene_color_encoder']['input_buffer']})
            cache_output['pred_scene_color_no_st'] = self.cur_output['pred_scene_color_no_st'].detach()
        for he_id in range(num_he-1):
            cache_output[f'{self.model.get_net().he_pfs[he_id]}sc_layers'] = [item.detach() for item in self.cur_output[f'{self.model.get_net().he_pfs[he_id]}sc_layers']]
            cache_output[f'{self.model.get_net().he_pfs[he_id]}output'] = self.cur_output[f'{self.model.get_net().he_pfs[he_id]}output'].detach()
        if self.model.get_net().enable_recurrent_lmv: 
            cache_output['pred_recurrent_tmv'] = self.cur_output['pred_recurrent_tmv'].detach()
            cache_output['merged_motion_vector_0'] = self.cur_data['merged_motion_vector_0'].detach()
            if net.enable_st_lmv or net.enable_st_lmv_res:
                cache_output['pred_recurrent_st_tmv'] = self.cur_output['pred_recurrent_st_tmv'].detach()
        if self.model.get_net().enable_recurrent_e2e:
            cache_output['e2e_sc_layers'] = [item.detach() for item in self.cur_output[f'{self.model.get_net().he_pfs[0]}e2e_sc_layers']]
        if self.model.get_net().enable_recurrent_d2e:
            cache_output['d2e_sc_layers'] = [item.detach() for item in self.cur_output['d2e_sc_layers']]
            cache_output['ge_sc_layers'] = [item.detach() for item in self.cur_output['ge_sc_layers']]
        if self.model.get_net().enable_recurrent_d2d:
            cache_output['d2d_sc_layers'] = [item.detach() for item in self.cur_output['d2d_sc_layers']]
            
        cache_output['metadata'] = copy.deepcopy(self.cur_data['metadata']) # type: ignore
        self.last_output.append(cache_output)
        
class ShadeNetV6SeqTrainer(ShadeNetV6Trainer):
    def __init__(self, args, model, resume=False):
        super().__init__(args, model, resume)
        self.output_cache = None
        self.last_output = []

    def update(self, data, epoch_index=None, batch_index=None, mode="train"):
        self.last_output = []
        for i, item in enumerate(data):
            if mode == 'train':
                self.cur_data_index = i
            else:
                self.cur_data_index += 1
            self.load_data(item, mode)
            self._update_one_batch(epoch_index=self.epoch_index, batch_index=self.batch_index, mode=mode)
        self.last_output.clear()