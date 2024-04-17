import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'TRUE'