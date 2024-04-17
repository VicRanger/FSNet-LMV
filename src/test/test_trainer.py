from copy import deepcopy
import copy
import datetime
import time
import torch
import argparse
import sys
import os
import includes.importer
from utils.str_utils import dict_to_string
from utils.config_enhancer import enhance_buffer_config, enhance_train_config, update_config
from utils.parser_utils import create_json_parser, create_py_parser
from utils.log import add_prefix_to_log, log, shutdown_log
from trainers.shade_net_trainer import ShadeNetTrainer, ShadeNetV5d4Trainer
import torch.distributed as dist
import torch.distributed
from models.shade_net.shade_net_v5d4 import ShadeNetModelV5d4
from config.config_utils import create_config, parse_config, convert_to_dict


def train(config_train):
    resume = config_train['args'].get('resume', False)
    model = eval(config_train['model']['class'])(
        config_train)
    trainer = eval(config_train['trainer']['class'])(
        config_train, model, resume=resume)
    trainer.train()


def test(config_test):
    log.debug("start test")
    config_test['use_ddp'] = False
    test_only = config['args']['test_only']
    resume = True
    if test_only:
        resume = False
    model = eval(config_test['model']['class'])(
        config_test)
    trainer = eval(config_test['trainer']['class'])(
        config_test, model, resume=resume)
    trainer.test()


def single_start(local_rank: int, config: dict) -> None:
    log.info("creating trainer, local_rank: {}".format(local_rank))
    log.debug("torch cuda gpu num: {}".format(torch.cuda.device_count()))

    if config['use_ddp'] and config['args']['train']:
        env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
        }
        log.debug(
            f"[{os.getpid()}] Initializing process group with: {env_dict}")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl',
                                init_method="env://",
                                )
        dist.barrier()
        log.debug(f"torch.distributed.is_initialized():{torch.distributed.is_initialized()}")
        config['local_rank'] = local_rank
        config['device'] = str(torch.device("cuda", config['local_rank']))
        log.debug("env_local_rank: {}, dist_local_rank:{}".format(
            local_rank, torch.distributed.get_rank()))
    else:
        config['local_rank'] = local_rank

    config_train = deepcopy(config)
    if config['args']['train']:
        train(config_train)
        if config_train['use_ddp']:
            dist.destroy_process_group()

    if config['args']['test'] and local_rank <= 0:
        config_test = deepcopy(config)
        if config['args']['train'] and 'time_string' in config_train.keys():
            config_test['time_string'] = config_train['time_string']
        test(config_test)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trainer")
    parser.add_argument("--config", help="trainer config file path")
    parser.add_argument("--num_gpu", default=0, type=int, help="num_gpu")
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    args = parser.parse_args()

    config = parse_config(args.config, root_path="")
    input_config = copy.deepcopy(convert_to_dict(create_config(args.config)))
    config['_input_config'] = input_config


    config['args'] = vars(args)
    update_config(config)
    enhance_train_config(config)

    log.debug("{}:\n {}".format(
        config['job_name'], dict_to_string(config['args'], 'args')))
    if (args.num_gpu) > 0:
        config['num_gpu'] = args.num_gpu

    if config['use_ddp'] and config['args']['train']:
        config['world_size'] = config['num_gpu']
        single_start(int(os.environ['LOCAL_RANK']), config)
        # import cProfile
        # cProfile.run("single_start(int(os.environ['LOCAL_RANK']), config)", filename=f"result_augment_loader_{os.environ['LOCAL_RANK']}.out", sort="cumulative")
    else:
        single_start(-1, config)
        # import cProfile
        # cProfile.run("single_start(-1, config)", filename=f"240330.out", sort="cumulative")
