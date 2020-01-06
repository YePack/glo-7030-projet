import os

from src.semantic.utils.utils import readfile, savefile
from src.semantic.unet.generate_masks import create_labels_from_dir
from src.semantic.training_function import predict
from src.mapping.utils import create_mapping_data

import sys
from src.semantic import unet
sys.modules['src.unet'] = unet
net = readfile('unet')

from src.semantic import history
sys.modules['src.history'] = history
histo = readfile('history_cross')
#del sys.modules['src.unet']
#savefile(net, 'net_refactor')

TRAINING_FOLDER = 'data/train'
PATH_DATA = 'data/raw/'
#Create Setup
path_to = os.path.normpath(PATH_DATA + os.sep + os.pardir) + '/'
create_mapping_data(net, path_data=PATH_DATA, path_to=path_to, train_test_perc=0.8, train_valid_perc=0.8, max=2)


