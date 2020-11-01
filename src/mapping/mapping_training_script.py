import os
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time

from src.semantic.utils.utils import readfile, savefile
from src.mapping.utils import create_mapping_data
from src.mapping.training_function import train_one_epoch, train_valid_loaders, validate_one_epoch
from src.mapping.model.homography_net import HomographyNet
from src.mapping.loss import MatchingLoss
from src.mapping.utils import get_device

import sys
from src.semantic import unet
sys.modules['src.unet'] = unet
net_semantic = readfile('unet_dice')

TRAINING_FOLDER = 'data/train/'
VALID_FOLDER = 'data/valid/'
PATH_DATA = 'data/raw/'


def train_homography_net(net, n_epoch, batch_size, lr):
    # Create Setup
    try:
        path_to = os.path.normpath(PATH_DATA + os.sep + os.pardir) + '/'
        create_mapping_data(net_semantic, path_data=PATH_DATA, path_to=path_to, train_test_perc=0.9,
                            train_valid_perc=0.9, max=40000)
    except FileExistsError:
        print('setup already created')

    model_dir = os.path.join('model_dir', datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir=model_dir)
    writer.add_text("parameters", "incoming parameters")

    device = get_device()
    criterion = MatchingLoss(device)
    net.to(device)

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    train_loader, val_loader = train_valid_loaders(train_path=TRAINING_FOLDER, valid_path=VALID_FOLDER,
                                                   batch_size=batch_size)
    for i in range(n_epoch):
        start = time.time()
        print('start epoch')
        running_loss = train_one_epoch(criterion, net, device, optimizer, train_loader)
        #val_loss = validate_one_epoch(net, device, val_loader, criterion, i, writer)
        val_loss = 999
        writer.add_scalar('Loss/train', running_loss, i + 1)
        writer.add_scalar('Loss/valid', val_loss, i+1)
        end = time.time()
        print(f'Epoch {i+1} - Train loss: {round(running_loss, 4)} - Val loss: {round(val_loss, 4)} Training time: {round(end-start,2)}s')
    writer.close()
    savefile(net, 'homo_net')

if __name__ == '__main__':
    homography_net = HomographyNet()
    train_homography_net(net=homography_net, n_epoch=30, batch_size=2, lr=0.0001)
