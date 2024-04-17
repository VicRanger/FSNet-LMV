
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import glob
import torch
from tqdm import tqdm
import includes.importer
from utils.warp import warp
from utils.dataset_utils import compose_scene_color, create_nov, create_scene_color_no_sky, create_st_color, data_to_device, get_input_filter_list
from dataloaders.dataset_base import dispatch_task_by_metadata
from dataloaders.patch_loader import PatchLoader
from utils.utils import remove_all_in_dir
from dataloaders.dataset_base import MetaData
from utils.dataset_utils import create_scene_color_no_st
from utils.config_enhancer import enhance_buffer_config
from config.config_utils import parse_config
from utils.loss_utils import psnr
from dataloaders.raw_data_importer import UE4RawDataLoader
from dataloaders.raw_data_importer import get_augmented_buffer
from utils.dataset_utils import create_de_color
from utils.str_utils import dict_to_string
from utils.loss_utils import psnr_hdr
from utils.utils import create_dir
from utils.log import log
from utils.buffer_utils import aces_tonemapper, align_channel_buffer, fix_dmdl_color_zero_value, write_buffer
from utils.dataset_utils import create_brdf_color
from utils.utils import get_tensor_mean_min_max_str

loader: PatchLoader
write_path, job_config = None, {}


def check_value(metadata: MetaData):
    global loader, write_path, job_config
    data = loader.load(metadata, history_config=history_config)
    data = data_to_device(data, "cuda")
    log.debug(f"check {metadata}")
    augmented_data_list = ['nov', 'brdf_color', 'dmdl_color', 'scene_light_no_st']
    [augmented_data_list.extend(job_config['buffer_config']['part'][part_name].get('augmented_data', []))
        for part_name in job_config['buffer_config']['basic_part_enable_list']]
    get_augmented_buffer(
        augmented_data_list,
        job_config['buffer_config'],
        data,
        last_data=[])
    
    
    data['scene_color_no_sky'] = create_scene_color_no_sky(data['scene_color_no_st'], data['sky_color'],data['skybox_mask'])
    data['scene_color_raw'] = data['scene_color'] + 0.0
    data['sky_color_raw'] = data['sky_color'] + 0.0
    data['scene_color_no_st_raw'] = data['scene_color_no_st'] + 0.0
    data['scene_color_no_sky_raw'] = data['scene_color_no_sky'] + 0.0

    data['comp_st_color'] = create_st_color(data['scene_color'],data['scene_color_no_st'], data['st_alpha'])
    data['debug_scene_color_minus_scene_color_no_st'] = data['scene_color'] - data['scene_color_no_st']
    ''' get exact scene_light for raw'''
    # ''' get sl_no_st for GT '''
    data['comp_scene_light_no_st'] = create_de_color(data['scene_color_no_st'], data['dmdl_color'],
                                                     data['skybox_mask'], data['sky_color'], fix=True)
    data['comp_scene_light_no_st'].clamp_(min=0.0, max=500.0)
    ''' get exact aa_scene_light for raw'''
    # ''' get sl_no_st for GT '''

    ''' regenerate sc for GT '''
    data['comp_scene_color'] = compose_scene_color(
        data['comp_scene_light_no_st'], data['dmdl_color'],
        data['st_color'], data['st_alpha'], data['skybox_mask'], data['sky_color'], fix=True)
    data['comp_scene_color_gt'] = data['scene_color_no_st'] * data['st_alpha'] + data['st_color']
    data['comp_scene_color_gt_light'] = compose_scene_color(
        data['scene_light_no_st'], data['dmdl_color'],
        data['st_color'], data['st_alpha'], data['skybox_mask'], data['sky_color'], fix=True)

    data[f'comp_brdf_color'] = create_brdf_color(
        data['roughness'], data['nov'], data['base_color'], data['metallic'], data['specular'], skybox_mask=data['skybox_mask'], fix=True)

    comp_sc_psnr = psnr_hdr(data['comp_scene_color'], data['scene_color'])
    log.debug(f"comp_sc_psnr: {comp_sc_psnr}")

    comp_brdf_psnr = psnr(data['comp_brdf_color'] * (1-data['skybox_mask']), data['brdf_color'] * (1-data['skybox_mask']))
    log.debug(f"comp_brdf_psnr: {comp_brdf_psnr}")
    illum_diff = ((aces_tonemapper(data['scene_color_no_st']) - aces_tonemapper(data['sky_color'])) * data['skybox_mask']).mean()
    log.debug(f'illum_diff: {illum_diff}')
    
    data['comp_history_warped_scene_color_no_st_0'] = warp(
        data['history_scene_color_no_st_0'], data['motion_vector'], padding_mode="border")[0]
    comp_hw_psnr = psnr_hdr(data['scene_color_no_st'], data['comp_history_warped_scene_color_no_st_0'])
    log.debug(f'comp_hw_psnr: {comp_hw_psnr}')
    # for k in data.keys():
    #     if not (isinstance(data[k], torch.Tensor)) or len(data[k].shape) != 3:
    #         continue
    #     if "motion_vector" in k or "cross_sample" in k:
    #         write_buffer("{}/{}_{}.exr".format(write_path, k, metadata.index),
    #                      data[k])
    #     else:
    #         write_buffer("{}/{}_{}.exr".format(write_path, k, metadata.index),
    #                      align_channel_buffer(
    #             data[k], channel_num=3, mode="repeat"))
    if comp_sc_psnr < 35 or\
        comp_brdf_psnr < 35 or\
        comp_hw_psnr < 15 or\
        illum_diff > 0.01:
        # data[f'aa_brdf_color'].min() < 0.01 - 1/65536:
        log.debug(dict_to_string(data, mmm=True))
        log.debug(metadata)
        # log.debug(data['metadata'])
        f = open("{}/output.log".format(write_path), 'a')
        f.write(f'{metadata}\tsc:{comp_sc_psnr:.2f},\tbrdf:{comp_brdf_psnr:.2f}\n')
        if comp_sc_psnr < 40 or data['scene_color_no_sky'].min()<0 or illum_diff > 0.01 or comp_brdf_psnr < 40:
            ks = ['dmdl_color', 'comp_scene_color', 'scene_color', 'st_color', 'comp_st_color', 'st_alpha', 'sky_color', 'scene_color_no_st',
                  'scene_color_raw', 'scene_color_no_st_raw', 'scene_color_no_sky_raw',
                  'comp_scene_light_no_st', 'comp_scene_color_gt_light', 'comp_scene_color_gt', 'scene_color_no_sky', 'debug_scene_color_minus_scene_color_no_st']
            for k in ks:
                if not 'raw' in k and not 'debug' in k and ('scene_color' in k or 'st_color' in k or 'sky_color' in k):
                    data[k] = aces_tonemapper(data[k])
                write_buffer("{}/{}_{}.exr".format(write_path, k, metadata.index),
                             align_channel_buffer(
                    data[k], channel_num=3, mode="repeat"))
        if comp_brdf_psnr < 40:
            ks = ['brdf_color', 'comp_brdf_color', 'roughness', 'nov', 'base_color', 'metallic', 'specular', 'skybox_mask']
            for k in ks:
                write_buffer("{}/{}_{}.exr".format(write_path, k, metadata.index),
                             align_channel_buffer(
                    data[k], channel_num=3, mode="repeat"))
        if comp_hw_psnr < 40:
            ks = ['history_scene_color_no_st_0', 'comp_history_warped_scene_color_no_st_0', 'scene_color_no_st']
            for k in ks:
                write_buffer("{}/{}_{}.exr".format(write_path, k, metadata.index),
                             align_channel_buffer(
                    data[k], channel_num=3, mode="repeat"))
        exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="exporter")
    parser.add_argument('--dual', action='store_true', default=False)
    parser.add_argument('--config', type=str, default="config/test/DT_60hz.yaml")
    args = parser.parse_args()
    config = {}
    job_config = parse_config("config/export/export_st.yaml")

    history_config = {
        'generate_from_data': False,
        'allow_skip': False,
        'num': 1,
        'part': [['metadata', 'scene_color_no_st']],
        'augmented_data': [[]]
    }
    require_list = []
    # history_config = config['dataset']['history_config']
    # require_filter = get_input_filter({
    #         'input_config': config,
    #         'input_buffer': config['model']['require_data']
    #     })
    enhance_buffer_config(job_config['buffer_config'],
                          history_num=history_config['num'])

    config['job_config'] = job_config
    config['buffer_config'] = job_config['buffer_config']
    loader = PatchLoader(
        config['buffer_config']['basic_part_enable_list'],
        buffer_config=config['buffer_config'],
        job_config=job_config,
        require_list=require_list,
        augmented_data_output=['history_scene_color_no_st_0'])

    # scene_name = "DT_T/DT_01_720"
    scene_name = job_config['scene'][0]
    for scene_name in job_config['scene']:
        log.debug("loading {}".format(scene_name))
        # start_index = 999
        start_index = 1
        # end_index = start_index + 1
        end_index = len(glob.glob(f'{job_config["export_path"]}/{scene_name}/metadata/*'))
        log.debug(f'{start_index}, {end_index}')
        write_path = "../output/images/check_value"
        create_dir(write_path)
        remove_all_in_dir(write_path)

        metadatas = [MetaData(scene_name, ind) for ind in range(start_index, end_index)]
        # for md in tqdm(metadatas):
        #     check_value(md)
        dispatch_task_by_metadata(check_value, metadatas,
                                num_thread=0)
