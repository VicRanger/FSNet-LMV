import logging
import time
from tqdm.auto import tqdm
import os

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)  

def get_local_rank():
    return int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ.keys() else 0
        
def add_prefix_to_log(prefix):
    global log
    class PrefixAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            time.sleep(0.001 + 0.01*get_local_rank())
            return f"{prefix}{msg}", kwargs
    log = PrefixAdapter(log, None)
    return log

def shutdown_log():
    global log, tdqm_stream
    log.removeHandler(tdqm_stream)

log = logging.getLogger("shading-flow-control")
formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
# sys_stream = logging.StreamHandler()
# sys_stream.setFormatter(formatter)
log.setLevel('DEBUG')
# log.addHandler(sys_stream)
tdqm_stream = TqdmLoggingHandler()
tdqm_stream.setFormatter(formatter)
log.addHandler(tdqm_stream)
log = add_prefix_to_log("LRK: {} - ".format(get_local_rank()))
