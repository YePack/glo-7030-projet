import numpy as np
import torch
import torch.nn as nn
import time
import glob
import math
import re
from torch.autograd import Variable
from torch.utils.data import DataLoader

from src.dataloader import DataGenerator
from src.net_parameters import p_number_of_classes


def train_valid_loaders(train_path, valid_path, batch_size, transform, shuffle=True):

    # List the files in train and valid
    train_images = glob.glob(train_path + '*.png')
    train_images.sort()
    train_labels_all = glob.glob(train_path + '*.pkl')
    train_labels_all.sort()
    train_labels_proportion = []
    train_labels = []
    for elem in train_labels_all:
        if re.search("_proportion", elem):
            train_labels_proportion.append(elem)
        else:
            train_labels.append(elem)

    valid_images = glob.glob(valid_path + '*.png')
    valid_images.sort()
    valid_labels_all = glob.glob(valid_path + '*.pkl')
    valid_labels_all.sort()
    valid_labels_proportion = []
    valid_labels = []
    for elem in valid_labels_all:
        if re.search("_proportion", elem):
            valid_labels_proportion.append(elem)
        else:
            valid_labels.append(elem)

    dataset_train = DataGenerator(train_images, train_labels, transform, train_labels_proportion)
    dataset_val = DataGenerator(valid_images, valid_labels, transform, valid_labels_proportion)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    return loader_train, loader_val


def validate(model, val_loader, criterion, use_gpu=False):

    model.train(False)
    val_loss = []
    val_loss_prop = []
    criterion_prop = nn.MSELoss()

    model.eval()

    for j, batch in enumerate(val_loader):

        inputs, targets, targets_prop = batch
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)
        output, output_prob = model(inputs)

        #predictions = output.max(dim=1)[1]

        val_loss.append(criterion(output, targets[:, 0]).item())
        val_loss_prop.append(criterion_prop(output_prob, targets_prop[:, 0]).item())
        #true.extend(targets.data.cpu().numpy().tolist())
        #pred.extend(predictions.data.cpu().numpy().tolist())

    model.train(True)
    #return accuracy_score(true, pred) * 100, sum(val_loss) / len(val_loss)
    return sum(val_loss) / len(val_loss), sum(val_loss_prop) / len(val_loss_prop)


def train(model, optimizer, train_path, valid_path, n_epoch, batch_size, transform, criterion, use_gpu=False,
          scheduler=None, shuffle=True, weight_adaptation=None):

    train_loader, val_loader = train_valid_loaders(train_path, valid_path, transform=transform,
                                                   batch_size=batch_size, shuffle=shuffle)

    for i in range(n_epoch):
        start = time.time()
        do_epoch(criterion, model, optimizer, scheduler, train_loader, use_gpu, weight_adaptation)

        train_loss = validate(model, train_loader, criterion, use_gpu)

        val_loss = validate(model, val_loader, criterion, use_gpu)
        end = time.time()

        print('Epoch {}  - Train loss: {:.4f} - Val loss: {:.4f} - Training time: {:.2f}s'.format(i,
                                                                                               train_loss[0],
                                                                                               val_loss[0],
                                                                                               end - start))

        print('Prop Loss - Train loss: {:.4f} - Val loss: {:.4f} - Training time: {:.2f}s'.format(i,
                                                                                               train_loss[1],
                                                                                               val_loss[1],
                                                                                               end - start))


def do_epoch(criterion, model, optimizer, scheduler, train_loader, use_gpu, weight_adaptation):
    model.train()
    if scheduler:
        scheduler.step()
    for batch in train_loader:

        inputs, targets, targets_prop = batch
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()
            targets_prop = targets_prop.cuda()

        inputs = Variable(inputs)
        targets = Variable(targets)
        targets_prop = Variable(targets_prop)
        optimizer.zero_grad()
        output, output_prop = model(inputs)

        if isinstance(criterion, torch.nn.modules.loss.CrossEntropyLoss):
            weight_learn = torch.FloatTensor(
                np.array([1/(np.log(1.1 + (np.array(targets.cpu() == i)).mean())) for i in range(p_number_of_classes)]))
            if weight_adaptation is not None:
                pred_unique = output.max(dim=1)[1].unique()
                targets_unique = targets.unique()
                for target in targets_unique:
                    if target not in pred_unique:
                        weight_learn[target] = weight_learn[target] + weight_adaptation
            if use_gpu:
                weight_learn = weight_learn.cuda()
            criterion = nn.CrossEntropyLoss(weight=weight_learn)
        criterion_prop = nn.MSELoss()
        loss = criterion(output, targets[:, 0])
        loss_prop = criterion_prop(output_prop, targets_prop[:, 0])
        loss_total = loss_prop + loss
        loss_total.backward()
        optimizer.step()
