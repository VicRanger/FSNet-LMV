
from utils.utils import create_dir, get_file_component
from yacs.config import CfgNode
from utils.str_utils import dict_to_string
from utils.log import log


def merge_from_another(a: CfgNode, b: CfgNode, allow_new=False):
    last_allow_new = a.is_new_allowed()
    if allow_new and not last_allow_new:
        a.set_new_allowed(allow_new)
    a.merge_from_other_cfg(b)

    if allow_new and not last_allow_new:
        a.set_new_allowed(last_allow_new)


def convert_to_dict(cfg_node):
    if not isinstance(cfg_node, CfgNode):
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v)
        return cfg_dict


def create_config(path: str) -> CfgNode:
    try:
        config = CfgNode.load_cfg(open(path, "r"))
    except Exception as e:
        log.debug(f"error when creating config {path}, {e}")
    else:
        return config


def parse_config(path: str, root_path="") -> CfgNode:
    config = create_config(root_path + path)
    # log.debug("{} config: ".format(path)+str(config.get('cuda_visible_devices', None)))
    # log.debug(type(config.get('cuda_visible_devices', None)))
    root_path = config.get("config_root_path", root_path)

    ''' add config as a dict's value.
        the key can be defined with \"include_name\".
    '''
    for tmp_path in config.get('include', []):
        tmp_cfg = parse_config(tmp_path, root_path=root_path)
        file_name = get_file_component(tmp_path)['filename']
        include_name = tmp_cfg.get('include_name', file_name)
        config[include_name] = tmp_cfg

    ''' add feature to current config.
        always with subclass, overwrite the base config.
    '''
    for tmp_path in config.get('pipeline', []):
        # log.debug(">===pipeline {} into {}:".format(tmp_path, path))
        tmp_cfg = parse_config(tmp_path, root_path=root_path)
        # log.debug("pip: {} config: ".format(path)+str(config.get('cuda_visible_devices', None)))
        # log.debug(type(config.get('cuda_visible_devices', None)))
        # log.debug("pip: {} tmp_cfg: ".format(tmp_path)+str(tmp_cfg.get('cuda_visible_devices', None)))
        # log.debug(type(tmp_cfg.get('cuda_visible_devices', None)))
        merge_from_another(tmp_cfg, config, allow_new=True)
        config = tmp_cfg
        # log.debug("{} pipline {} config: ".format(path, tmp_path)+str(config.get('cuda_visible_devices', None)))
        # log.debug(type(config.get('cuda_visible_devices', None)))

    ''' import baseline config.
        only one base file allowed.
    '''
    tmp_path = config.get('base', None)
    if tmp_path is not None:
        # log.debug(">===base {} into {}:".format(tmp_path, path))
        tmp_cfg = parse_config(tmp_path, root_path=root_path)
        # log.debug("{} config: ".format(path)+str(config.get('cuda_visible_devices', None)))
        # log.debug(type(config.get('cuda_visible_devices', None)))
        # log.debug("{} tmp_cfg: ".format(tmp_path)+str(tmp_cfg.get('cuda_visible_devices', None)))
        # log.debug(type(tmp_cfg.get('cuda_visible_devices', None)))
        merge_from_another(tmp_cfg, config, allow_new=True)
        config = tmp_cfg

    return config
