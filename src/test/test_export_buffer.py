import argparse
import includes.importer
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# from utils.utils import *
from utils.config_enhancer import enhance_buffer_config
from dataloaders.dataset_base import dispatch_task_by_metadata
from dataloaders.raw_data_importer import UE4RawDataLoader
from utils.log import log
from utils.str_utils import dict_to_string
from config.config_utils import parse_config

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="exporter")
    parser.add_argument('--config', type=str, default="config/export/export_st.yaml")
    # parser.add_argument('--config', type=str, default="config/export/export_no_st.yaml")
    args = parser.parse_args()
    job_config = parse_config(args.config)

    # log.debug(dict_to_string(job_config))
    if 'dataset' in job_config.keys():
        scale_config = job_config['dataset'].get('scale_config', {})
    else:
        scale_config = {}
    enhance_buffer_config(job_config['buffer_config'], scale_config=scale_config)
    raw_data_loader = UE4RawDataLoader(job_config['buffer_config'], job_config)
    # log.setLevel('DEBUG')
    # raw_data_loader.export_patch("DT/DT_01", 1, 1)
    raw_data_loader.export()
