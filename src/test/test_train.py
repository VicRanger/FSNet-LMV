import argparse
import subprocess
import os
import sys
import platform

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


if __name__ == "__main__":
    from utils.parser_utils import create_py_parser
    from config.config_utils import create_config, parse_config, convert_to_dict
    from utils.log import log
    from utils.str_utils import dict_to_string
    parser = argparse.ArgumentParser(description="trainer")
    parser.add_argument("--config", help="trainer config file path")
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--multi', action='store_true', default=False)
    args = parser.parse_args()
    config_file = args.config

    program = 'src/test/test_trainer.py'
    config = parse_config(config_file, root_path="")
    num_gpu = config['num_gpu']
    port = find_free_port()

    prefix = ""
    type_input = str(config['cuda_visible_devices'])
    type_input = type_input.replace(" ", "").replace("(", "").replace(")", "").replace("\\", "")
    os.environ['CUDA_VISIBLE_DEVICES'] = type_input
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    train = args.train
    test = args.test
    test_only = args.test_only
    resume = train and args.resume
    # debug_info = "export TORCH_DISTRIBUTED_DEBUG=INFO;"
    debug_info = ""
    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    command_str = None
    if train:
        config_str = "--config {} {} {}".format(config_file,
                                                   "--train" if train else "",
                                                   "--resume" if resume else "")
                                                #    f"--port {port}" if num_gpu > 1 else "")
        if num_gpu > 1:
            bash_command = f"torchrun \
--rdzv_backend=c10d \
--rdzv_endpoint=localhost:{port} \
--nnode=1 \
--nproc_per_node={num_gpu} \
{program} {config_str}"
        else:
            bash_command = "python {} --num_gpu {} {}".format(
                program, num_gpu, config_str)
        command_str = "{} {} {}".format(debug_info, prefix, bash_command)
    if test:
        config_str = "--config {} {} {} {}".format(config_file,
                                                   "--resume" if resume else "",
                                                   "--test" if test else "",
                                                   "--test_only" if test_only else "")
        bash_command = "python {} --num_gpu {} {}".format(
            program, 1, config_str)
        if command_str is not None:
            command_str = command_str + " && " + "{} {} {}".format(debug_info, prefix, bash_command)
        else:
            command_str = "{} {} {}".format(debug_info, prefix, bash_command)

    log.debug("run command: {}".format(command_str))
    assert command_str is not None
    subprocess.run(command_str, shell=True)
