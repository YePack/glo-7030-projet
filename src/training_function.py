import numpy as np
import torch
import torch.nn as nn
import time
import glob
import math
from torch.autograd import Variable
from torch.utils.data import DataLoader

from src.dataloader import DataGenerator
from src.net_parameters import p_number_of_classes
from src.history import History


def train_valid_loaders(train_path, valid_path, batch_size, transform, shuffle=True):

    # List the files in train and valid
    train_images = glob.glob(train_path + '*.png')
    train_labels = glob.glob(train_path + '*.pkl')
    valid_images = glob.glob(valid_path + '*.png')
    valid_labels = glob.glob(valid_path + '*.pkl')

    dataset_train = DataGenerator(train_images, train_labels, transform)
    dataset_val = DataGenerator(valid_images, valid_labels, transform)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    return loader_train, loader_val


def validate(model, val_loader, criterion, use_gpu=False):

    model.train(False)
    val_loss = []

    model.eval()

    for j, batch in enumerate(val_loader):

        inputs, targets = batch
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)
        output = model(inputs)

        #predictions = output.max(dim=1)[1]

        val_loss.append(criterion(output, targets[:, 0]).item())
        #true.extend(targets.data.cpu().numpy().tolist())
        #pred.extend(predictions.data.cpu().numpy().tolist())

    model.train(True)
    #return accuracy_score(true, pred) * 100, sum(val_loss) / len(val_loss)
    return sum(val_loss) / len(val_loss)


def train(model, optimizer, train_path, valid_path, n_epoch, batch_size, transform, criterion, use_gpu=False,
          scheduler=None, shuffle=True, weight_adaptation=None):
    history = History()

    train_loader, val_loader = train_valid_loaders(train_path, valid_path, transform=transform,
                                                   batch_size=batch_size, shuffle=shuffle)

    for i in range(n_epoch):
        start = time.time()
        do_epoch(criterion, model, optimizer, scheduler, train_loader, use_gpu, weight_adaptation)

        train_loss = validate(model, train_loader, criterion, use_gpu)

        val_loss = validate(model, val_loader, criterion, use_gpu)
        end = time.time()
        history.save(train_loss, val_loss, optimizer.param_groups[0]['lr'])

        print('Epoch {} - Train loss: {:.4f} - Val loss: {:.4f} Training time: {:.2f}s'.format(i,
                                                                                               train_loss,
                                                                                               val_loss,
                                                                                               end - start))
    return history


def do_epoch(criterion, model, optimizer, scheduler, train_loader, use_gpu, weight_adaptation):
    model.train()
    if scheduler:
        scheduler.step()
    for batch in train_loader:

        inputs, targets = batch
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = Variable(inputs)
        targets = Variable(targets)
        optimizer.zero_grad()
        output = model(inputs)

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

        loss = criterion(output, targets[:, 0])
        loss.backward()
        optimizer.step()
