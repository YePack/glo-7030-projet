import numpy as np
import torch
import torch.nn as nn
import time
import glob
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from src.mapping.dataloader import DataGenerator
from src.semantic.history import History


def train_valid_loaders(train_path, valid_path, batch_size, shuffle=True):
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

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    return loader_train, loader_val


def validate(model, val_loader, criterion, use_gpu=False):
    # TODO adapt for mapping
    model.train(False)
    val_loss = []

    model.eval()

    for j, batch in enumerate(val_loader):

        inputs, targets = batch
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        #inputs = Variable(inputs, volatile=True)
        #targets = Variable(targets, volatile=True)
        output = model(inputs)

        # predictions = output.max(dim=1)[1]

        val_loss.append(criterion(output, targets[:, 0]).item())
        # true.extend(targets.data.cpu().numpy().tolist())
        # pred.extend(predictions.data.cpu().numpy().tolist())

    model.train(True)
    # return accuracy_score(true, pred) * 100, sum(val_loss) / len(val_loss)
    return sum(val_loss) / len(val_loss)


def train(model, optimizer, train_path, valid_path, n_epoch, batch_size, criterion, use_gpu=False,
          scheduler=None, shuffle=True,):
    # TODO adapt for mapping
    history = History()

    train_loader, val_loader = train_valid_loaders(train_path, valid_path, batch_size=batch_size, shuffle=shuffle)

    for i in range(n_epoch):
        start = time.time()
        do_epoch(criterion, model, optimizer, scheduler, train_loader, use_gpu)

        train_loss = validate(model, train_loader, criterion, use_gpu)

        val_loss = validate(model, val_loader, criterion, use_gpu)
        end = time.time()
        history.save(train_loss, val_loss, optimizer.param_groups[0]['lr'])

        print('Epoch {} - Train loss: {:.4f} - Val loss: {:.4f} Training time: {:.2f}s'.format(i,
                                                                                               train_loss,
                                                                                               val_loss,
                                                                                               end - start))
    return history


def do_epoch(criterion, model, optimizer, scheduler, train_loader, use_gpu):
    # TODO adapt for mapping
    model.train()
    if scheduler:
        scheduler.step()
    for batch in train_loader:

        inputs, labels = batch
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        #inputs = Variable(inputs)
        #targets = Variable(targets)
        optimizer.zero_grad()
        output_h = model(inputs)

        loss = criterion(labels, h)
        loss.backward()
        optimizer.step()


def predict(model, image_path, folder):
    # TODO adapt for mapping
    model.eval()
    pass
