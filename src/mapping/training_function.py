import numpy as np
import torch
import torch.nn as nn
import time
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms


from src.mapping.dataloader import DataGenerator
from src.mapping.utils import warpPerspective, show_projected_label


def train_valid_loaders(train_path, valid_path, batch_size):
    # List the files in train and valid
    train_images = glob.glob(train_path + '*_semantic.pkl')
    train_labels = glob.glob(train_path + '*[0-9].*pkl')
    valid_images = glob.glob(valid_path + '*_semantic.pkl')
    valid_labels = glob.glob(valid_path + '*[0-9].*pkl')

    train_images.sort()
    train_labels.sort()
    valid_images.sort()
    valid_labels.sort()

    dataset_train = DataGenerator(train_images, train_labels)
    dataset_val = DataGenerator(valid_images, valid_labels)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    return loader_train, loader_val


def validate_one_epoch(model, device, val_loader, criterion, n_epoch, writer):
    model.train(False)
    val_loss = []
    classes_color = ['black', 'white', 'yellow', 'pink', 'coral', 'crimson', 'blue', 'red', 'magenta', 'green']
    cmap = mpl.colors.ListedColormap(classes_color)

    model.eval()
    image_print = 0
    for j, batch in enumerate(val_loader):

        inputs, labels, image_name = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        output_h = model(inputs)
        val_loss.append(criterion(labels, output_h).item())

        if image_print < 4:
            for i in range(len(inputs)):
                h = output_h[i].reshape(3, 3)
                projected_label = warpPerspective(labels[i].unsqueeze(0), h.unsqueeze(0), device)
                plt.imshow(projected_label[0].cpu(), cmap=cmap)
                fig = plt.gcf()
                writer.add_figure(f"{image_name[i]}", fig, n_epoch+1)
                image_print += 1

    # return accuracy_score(true, pred) * 100, sum(val_loss) / len(val_loss)
    return sum(val_loss) / len(val_loss)


def train_one_epoch(criterion, model, device, optimizer, train_loader):
    model.train()
    running_loss = []
    for i, batch in enumerate(train_loader):
        inputs, labels, _ = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output_h = model(inputs)
        loss = criterion(labels, output_h)
        running_loss.append(loss)
        loss.backward()
        optimizer.step()
    return np.array(running_loss).mean()


def predict(model, image_path, folder):
    # TODO adapt for mapping
    model.eval()
    pass
