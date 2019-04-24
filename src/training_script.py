import torch
import torch.nn as nn
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt

from optparse import OptionParser
from torch import optim
from src.unet.unet_model import UNet

from src import train
from src.dataloader import DataGenerator
from src.dataloader import NormalizeCropTransform
from src.unet.generate_masks import create_labels_from_dir
from src.dataloader.flip_images import flip_images


def train_unet(net, path_train, path_valid, n_epoch, batch_size, lr, criterion, use_gpu):

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    transform = NormalizeCropTransform(normalize=True, crop=(450, 256))

    train(model=net, optimizer=optimizer, train_path=path_train, valid_path=path_valid, n_epoch=n_epoch,
          batch_size=batch_size, criterion=criterion, transform=transform, use_gpu=use_gpu, weight_adaptation=None)


def see_image_output(net, path_img, path_xml, transform):
    # {'crowd': 0, 'ice': 1, 'board': 2, 'circlezone': 3, 'circlemid': 4, 'goal': 5, 'blue': 6, 'red': 7, 'fo': 8}
    colors = ['black', 'white', 'yellow', 'pink', 'coral', 'crimson', 'blue', 'red', 'magenta']
    cmap = mpl.colors.ListedColormap(colors)
    net.eval()
    data = DataGenerator(path_img, path_xml, transform=transform)
    i = 0
    while i < len(data):
        fig, subfigs = plt.subplots(2, 2)
        for j, subfig in enumerate(subfigs.reshape(-1)):
            if j % 2 == 0:
                img, label = data[i]
                img.unsqueeze_(0)
                preds = net(img)
                preds_img = preds.max(dim=1)[1]
                subfig.imshow(preds_img[0], cmap=cmap)
            else:
                subfig.imshow(label[0], cmap=cmap)
                i += 1

        plt.show()

# See the train prediction
#see_image_output(net, train_images, path_xml_train, transform)

# See the valid prediction
#see_image_output(net, path_img_val, path_xml_val, transform)


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
    parser.add_option('-m', '--load', dest='load',
                      default=False, help='load file model')
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
    else:
        sys.exit(0)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()

    if args.setup:
        # Split train and test in 2 different folders (and save arrays instead of XMLs)
        path_to = os.path.normpath(args.path + os.sep + os.pardir)+'/'
        create_labels_from_dir(path_data=args.path, path_to=path_to, train_test_perc=0.8, train_valid_perc=0.8)
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
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

