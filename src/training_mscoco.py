import os
import sys
from optparse import OptionParser

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch import optim
from src.unet.unet_model import UNet
from src.training_function import train_mscoco
from src.dataloader import NormalizeCropTransform
from src.unet.utils import readfile, savefile
from src.utils.show_images_sample import see_image_output

from src.net_parameters import p_weight_augmentation, p_normalize, p_model_name_save

def train_mscoco_unet(net, path_data, n_epoch, batch_size, lr, criterion, use_gpu):

    """ This function trains a unet model from 2014 COCO dataset

    Args:
        - net: the neural network architecture
        - path_data: path where COCO dataset is saved. This path needs to have 
          the followings subfolders: 
            images/train2014/: train images from 2014 MSCOCO
            images/valid2014/: train images from 2014 MSCOCO
            annotations/instances_train2014.json: train annotations file from 2014 MSCOCO
            annotations/instances_valid2014.json: train annotations file from 2014 MSCOCO
        - n_epochs: number of epochs to run
        - batch_size: batch size
        - lr: learning rate
        - criterion: loss criterion (torch.nn loss)
        - use_gpu: if True the GPU will be used
    """

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    transform = NormalizeCropTransform(normalize=p_normalize, crop=(450, 256))

    train_mscoco(model=net, optimizer=optimizer, path_data=path_data, n_epoch=n_epoch, 
                batch_size=batch_size, criterion=criterion, use_gpu=use_gpu,
                transform=transform, weight_adaptation=p_weight_augmentation)

    net.cpu()
    savefile(net, p_model_name_save)


def get_args():
    parser = OptionParser()
    parser.add_option('-p', '--path', type=str, dest='path', default='data/mscoco/',
                      help='Path COCO raw data (images and annotations)')
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=2,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use gpu')
    parser.add_option('-n', '--model_load_name', type=str, dest='model_name', default='',
                      help='model dict saved')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    
    args = get_args()

    net = UNet(3, 80)

    if args.model_name != '':
        net = net.load_state_dict(torch.load(args.model_name))
        print(f'Model loaded from this dict : {args.model_name}')

    if args.gpu:
        net.cuda()

    criterion = nn.CrossEntropyLoss()

    try:
        train_mscoco_unet(net=net,
                          path_data=args.path,
                          n_epoch=args.epochs,
                          batch_size=args.batchsize,
                          lr=args.lr,
                          criterion=criterion,
                          use_gpu=args.gpu)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), os.path.join('.', args.model_name))
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

