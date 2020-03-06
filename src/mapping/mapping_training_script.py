import os
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from src.semantic.utils.utils import readfile, savefile
from src.semantic.unet.generate_masks import create_labels_from_dir
from src.semantic.training_function import predict
from src.mapping.utils import create_mapping_data
from src.mapping.training_function import train
from src.mapping.model.homography_net import HomographyNet
from src.mapping.loss import MatchingLoss

import sys
from src.semantic import unet
sys.modules['src.unet'] = unet
net_semantic = readfile('unet_dice')

#from src.semantic import history
#sys.modules['src.history'] = history
#histo = readfile('history_cross')
#del sys.modules['src.unet']
#savefile(net, 'net_refactor')

TRAINING_FOLDER = 'data/train/'
VALID_FOLDER = 'data/valid/'
PATH_DATA = 'data/raw/'
#Create Setup
path_to = os.path.normpath(PATH_DATA + os.sep + os.pardir) + '/'
#create_mapping_data(net_semantic, path_data=PATH_DATA, path_to=path_to, train_test_perc=0.9, train_valid_perc=0.9, max=10)


def train_homography_net(net, path_train, path_valid, n_epoch, batch_size, lr, criterion, use_gpu):

    model_dir = os.path.join('model_dir', datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir=model_dir)
    writer.add_text("parameters", "incoming parameters")
    if use_gpu:
        net.cuda()
    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    history = train(model=net, model_dir=model_dir, optimizer=optimizer, train_path=path_train, valid_path=path_valid, n_epoch=n_epoch,
          batch_size=batch_size, criterion=criterion, use_gpu=use_gpu, writer=writer)
    net.cpu()
    savefile(net, 'homo_net')
    return history

if __name__ == '__main__':
    net = HomographyNet()
    train_homography_net(net, TRAINING_FOLDER, VALID_FOLDER, 10, 2, 0.0001, MatchingLoss(), True)
