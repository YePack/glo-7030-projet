import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import numpy as np
import re

from src.dataloader.transform import NormalizeCropTransform
from src.dataloader.dataset import DataGenerator
from src.net_parameters import p_classes_color


def see_image_output(net, path_train, path_test, path_save):
    cmap = mpl.colors.ListedColormap(p_classes_color)
    net.cpu()
    net.eval()
    transform = NormalizeCropTransform(normalize=True, crop=(450, 256))

    # Sample 2 images from train
    train_images = glob.glob(path_train + '*.png')
    nb_images = len(train_images)
    indices = np.arange(nb_images)
    np.random.shuffle(indices)
    imgs_train = [train_images[i] for i in indices[:2]]
    imgs_train.sort()
    labels_train = [s for s in glob.glob(path_train + '*.pkl') if any(xs in s for xs in ['/'+i.split('/')[-1].replace('.png', '') for i in imgs_train])]
    labels_train.sort()

    for i, label in enumerate(labels_train):
        if re.search('proportion', label):
            labels_train.pop(i)


    # Sample 2 images from test
    test_images = glob.glob(path_test + '*.png')
    nb_images = len(test_images)
    indices = np.arange(nb_images)
    np.random.shuffle(indices)
    imgs_test = [test_images[i] for i in indices[:2]]
    imgs_test.sort()
    labels_test = [s for s in glob.glob(path_test + '*.pkl') if any(xs in s for xs in ['/'+i.split('/')[-1].replace('.png', '') for i in imgs_test])]
    labels_test.sort()

    for i, label in enumerate(labels_test):
        if re.search('proportion', label):
            labels_test.pop(i)



    print('Train: '+str(imgs_train)+str(labels_train))
    print('Test: ' + str(imgs_test) + str(labels_test))

    data_train = DataGenerator(imgs_train, labels_train, transform=transform)
    data_test = DataGenerator(imgs_test, labels_test, transform=transform)

    i = 0
    while i < len(data_train):
        fig, subfigs = plt.subplots(2, 2)
        for j, subfig in enumerate(subfigs.reshape(-1)):
            if j % 2 == 0:
                img, label, _ = data_train[i]
                img.unsqueeze_(0)
                preds, _ = net(img)
                preds_img = preds.max(dim=1)[1]
                subfig.imshow(preds_img[0], cmap=cmap)
                subfig.set_title('Predictions #'+str(i+1))
            else:
                subfig.imshow(label[0], cmap=cmap)
                subfig.set_title('Ground truth #' + str(i + 1))
                i += 1

        fig.suptitle("Sample predicted from train dataset", fontsize=16, y=1.002, x=0.4)
        plt.savefig(path_save+'train-sample.png')
        plt.show()

    i = 0
    while i < len(data_test):
        fig, subfigs = plt.subplots(2, 2)
        for j, subfig in enumerate(subfigs.reshape(-1)):
            if j % 2 == 0:
                img, label, _ = data_test[i]
                img.unsqueeze_(0)
                preds, _ = net(img)
                preds_img = preds.max(dim=1)[1]
                subfig.imshow(preds_img[0], cmap=cmap)
                subfig.set_title('Predictions #' + str(i + 1))
            else:
                subfig.imshow(label[0], cmap=cmap)
                subfig.set_title('Ground truth #' + str(i + 1))
                i += 1

        fig.suptitle("Sample predicted from test dataset", fontsize=16, y=1.002, x=0.4)
        plt.savefig(path_save+'test-sample.png')
        plt.show()
