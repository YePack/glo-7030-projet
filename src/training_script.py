import torch.nn as nn
import os
import sys

from optparse import OptionParser
from torch import optim
from src.unet.unet_model import UNet

from src import train
from src.dataloader import NormalizeCropTransform
from src.loss import DiceCoeff
from src.unet.generate_masks import create_labels_from_dir
from src.dataloader.flip_images import flip_images
from src.create_image_label.show_images_sample import see_image_output
from src.unet.utils import readfile, savefile
from src.net_parameters import p_weight_augmentation, p_normalize, p_model_name_save, max_images


def train_unet(net, path_train, path_valid, n_epoch, batch_size, lr, criterion, use_gpu):

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)


    transform = NormalizeCropTransform(normalize=p_normalize, crop=(450, 256))

    train(model=net, optimizer=optimizer, train_path=path_train, valid_path=path_valid, n_epoch=n_epoch,
          batch_size=batch_size, criterion=criterion, transform=transform, use_gpu=use_gpu,
          weight_adaptation=p_weight_augmentation)
    net.cpu()
    savefile(net, p_model_name_save)

def get_args():
    parser = OptionParser()
    parser.add_option('-p', '--path', type=str, dest='path', default='data/raw/',
                      help='Path raw data (.png and .xml)')
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=2,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-c', '--criterion', type=str, dest='criterion', default='CrossEntropy',
                      help='Choices: CrossEntropy or Dice')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use gpu')
    parser.add_option('-m', '--model', type=str, dest='model', default='',
                      help='Model to load (path to the pickle)')
    parser.add_option('-s', '--setup', dest='setup', action='store_true',
                      default=False, help='Setup the datasets otpion.')
    parser.add_option('-a', '--augmentation', dest='augmentation', action='store_true',
                      default=False, help='data augmentation option. Need to have set up to true.')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    net = UNet(3, 9)

    if args.criterion == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == 'Dice':
        criterion = DiceCoeff()
    else:
        sys.exit(0)

    if args.model != '':
        net = readfile(args.model)
        print('Model loaded from this pickle : {}'.format(args.model))

    if args.gpu:
        net.cuda()

    # We assume the path to save is the path parent to the raw/ data
    path_to = os.path.normpath(args.path + os.sep + os.pardir) + '/'
    if args.setup:
        # Split train and test in 2 different folders (and save arrays instead of XMLs)
        create_labels_from_dir(path_data=args.path, path_to=path_to, train_test_perc=0.8, train_valid_perc=0.8,
                               max=max_images)
        if args.augmentation:
            flip_images(path_to+'train/')

    try:
        train_unet(net=net,
                   path_train=path_to+'train/',
                   path_valid=path_to+'valid/',
                   n_epoch=args.epochs,
                   batch_size=args.batchsize,
                   lr=args.lr,
                   use_gpu=args.gpu,
                   criterion=criterion)

        see_image_output(net, path_train=path_to+'train/', path_test=path_to+'test/', path_save=path_to)
    except KeyboardInterrupt:
        savefile(net, p_model_name_save)
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

