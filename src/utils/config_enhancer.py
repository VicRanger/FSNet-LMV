import copy
import json
from utils.dataset_utils import get_input_filter_list
from utils.parser_utils import parse_buffer_name
from utils.str_utils import dict_to_string
from utils.utils import del_dict_item
from utils.log import log

default_scale_config = {"ds_scale": 2}


def initialize_recipe(recipe: dict, history_num: int = 0, history_list: list[int] = [0, 1, 2, 4], scale_config={}):
    ''' demodulation color buffer '''
    albedo_name = "brdf_color" if recipe.get("brdf_demodulate", False) else "base_color"
    for buffer_name in recipe['augmented_data_recipe__demodulate_template']:
        target = buffer_name.replace("scene_color", "scene_light")
        recipe['augmented_data_recipe'][target] = {
            "dep": [buffer_name, albedo_name],
        }

    ''' history dupilication '''
    history_num = history_num if history_num > 0 else recipe.get("augmented_data_recipe__history_num", history_num)
    for buffer_name in recipe["augmented_data_recipe__history_template"]:
        dep_history = []
        for i in range(history_num):
            if i in history_list:
                dep_history.append([buffer_name])
            else:
                dep_history.append([])
            recipe['augmented_data_attribute']['history_' + buffer_name + f"_{i}"] = recipe['augmented_data_attribute'][buffer_name]
            recipe['augmented_data_recipe']["history_" + buffer_name + f"_{i}"] = {
                "dep": [],
                "num_history": history_num,
                "dep_history": [[buffer_name] if _ == i else [] for _ in range(history_num)]
            }
        recipe['augmented_data_attribute']['history_' + buffer_name] = recipe['augmented_data_attribute'][buffer_name]
        recipe['augmented_data_recipe']["history_" + buffer_name] = {
            "dep": [],
            "num_history": history_num,
            "dep_history": dep_history
        }
        
        
    ''' merged_motion_vector '''
    recipe['augmented_data_recipe']['merged_motion_vector']['num_history'] = history_num

    ''' history warping '''
    for buffer_name in recipe["augmented_data_recipe__history_warped_template"]:
        dep_history = []
        dep = []
        for i in range(history_num):
            if i in history_list:
                dep_history.append([buffer_name])
                dep.append(f"merged_motion_vector_{i}")
            else:
                dep_history.append([])
        recipe['augmented_data_attribute']['history_warped_' + buffer_name] = recipe['augmented_data_attribute'][buffer_name]
        recipe['augmented_data_recipe']["history_warped_" + buffer_name] = {
            "dep": dep,
            "num_history": history_num,
            "dep_history": dep_history
        }

    ''' ssaa prefix dupilication'''
    log.debug(dict_to_string(scale_config))
    attr = recipe['augmented_data_attribute']
    old_recipe = copy.deepcopy(recipe['augmented_data_recipe'])
    additional_recipe = {}
    for _, config in scale_config.items():
        if not config['enable']:
            continue
        if config['pattern'] not in [r'%ds']:
            continue
        for buffer_name in old_recipe.keys():
            name = parse_buffer_name(buffer_name)['buffer_name']
            if attr[name]['type'] == 'image':
                new_item = copy.deepcopy(
                    recipe['augmented_data_recipe'][buffer_name])
                for index, value in enumerate(new_item["dep"]):
                    if attr[name]['type'] == 'image':
                        new_item['dep'][index] = f'{config["target"].format(config["value"])}_{value}'
                for i, dep in enumerate(new_item.get('dep_history', [])):
                    for j, value in enumerate(dep):
                        if attr[name]['type'] == 'image':
                            new_item['dep_history'][i][j] = f'{config["target"].format(config["value"])}_{value}'
                # log.debug(dict_to_string(new_item))
                additional_recipe[f'{config["target"].format(config["value"])}_{buffer_name}'] = new_item
    recipe['augmented_data_recipe'].update(additional_recipe)


def enhance_buffer_config(buffer_config, history_num=3, history_list=[], scale_config=None):
    if scale_config is None:
        scale_config = default_scale_config
    # log.debug(dict_to_string(scale_config))
    for name, value in scale_config.items():
        buffer_config[name] = value
        buffer_config["scale_regex"][name]['value'] = value
        buffer_config["scale_regex"][name]['enable'] = True

    ''' format the scale regex '''
    json_str = json.dumps(buffer_config)
    for scale_name, config in buffer_config["scale_regex"].items():
        if scale_name not in scale_config.keys():
            continue
        json_str = json_str.replace(config['pattern'], config['target'].format(config['value']))
    new_dict = json.loads(json_str)
    new_dict = del_dict_item(new_dict, "scale_regex")
    buffer_config.update(new_dict)

    # log.debug(dict_to_string(buffer_config))
    augmented_data_recipe = copy.deepcopy(buffer_config['augmented_data_recipe'])
    if len(history_list) <= 0:
        history_list = [i for i in range(history_num)]

    initialize_recipe(augmented_data_recipe,
                      history_num=history_num,
                      history_list=history_list,
                      scale_config=buffer_config['scale_regex'])
    # log.debug(dict_to_string(augmented_data_recipe['augmented_data_recipe']))
    buffer_config.update({'augmented_data_recipe': augmented_data_recipe['augmented_data_recipe']})
    # log.debug(dict_to_string(buffer_config['augmented_data_recipe']))


def enhance_train_config(config):
    config['dataset']['augment_loader'] = config['dataset'].get('augment_loader', True)
    input_buffer = config.get('model', {}).get('require_data', {})
    config['dataset']['require_list'] = get_input_filter_list({
        'input_config': config,
        'input_buffer': input_buffer
    })
    history_num = config['dataset']["history_config"]["num"]
    history_list = config['dataset']['history_config'].get("index", [i for i in range(history_num)])
    config['dataset']['history_config']["index"] = history_list
    if "demodulation_mode" in config['dataset'].keys():
        config['buffer_config']['demodulation_mode'] = config['dataset']['demodulation_mode']
        
    enhance_buffer_config(config['buffer_config'], history_num=history_num, history_list=history_list,
                          scale_config=config['dataset'].get('scale_config', {}))
    
    ''' set history_config to buffer_config '''
    config['buffer_config']['history_config'] = config['dataset'].get('history_config', None)
    
    ''' add export_path overwrite'''
    # config['dataset']['path'] = config['job_config']['export_path']
    
    
def update_config(config):
    config['use_ddp'] = config['num_gpu'] > 1
    config["use_cuda"] = config['num_gpu'] > 0
    config['device'] = "cuda:0" if config["use_cuda"] else "cpu"
    assert config['train_parameter']['batch_size'] % max(config['num_gpu'], 1) == 0
    config['train_parameter']['batch_size'] = config['train_parameter']['batch_size'] // max(config['num_gpu'], 1)
    assert config['dataset']['train_num_worker_sum'] % max(config['num_gpu'], 1) == 0
    config['dataset']['train_num_worker'] = config['dataset']['train_num_worker_sum'] // max(config['num_gpu'], 1)