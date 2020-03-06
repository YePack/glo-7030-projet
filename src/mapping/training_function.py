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


def validate(model, val_loader, criterion, n_epoch, writer, use_gpu=False):
    # TODO adapt for mapping
    model.train(False)
    val_loss = []
    classes_color = ['black', 'white', 'yellow', 'pink', 'coral', 'crimson', 'blue', 'red', 'magenta', 'green']
    cmap = mpl.colors.ListedColormap(classes_color)

    model.eval()

    for j, batch in enumerate(val_loader):

        inputs, labels, image_name = batch
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        #inputs = Variable(inputs, volatile=True)
        #targets = Variable(targets, volatile=True)
        output_h = model(inputs)

        # predictions = output.max(dim=1)[1]

        val_loss.append(criterion(labels, output_h).item())
        # true.extend(targets.data.cpu().numpy().tolist())
        # pred.extend(predictions.data.cpu().numpy().tolist())
        for i in range(len(inputs)):
            h = output_h[i].reshape(3, 3)
            projected_label = warpPerspective(labels[i].unsqueeze(0), h.unsqueeze(0))
            plt.imshow(projected_label[0], cmap=cmap)
            fig = plt.gcf()
            writer.add_figure(f"{image_name[i]}", fig, n_epoch+1)



    model.train(True)
    # return accuracy_score(true, pred) * 100, sum(val_loss) / len(val_loss)
    return sum(val_loss) / len(val_loss)


def train(model, model_dir, optimizer, train_path, valid_path, n_epoch, batch_size, criterion, use_gpu=False,
          scheduler=None, shuffle=True, writer=None):
    history = History()

    train_loader, val_loader = train_valid_loaders(train_path, valid_path, batch_size=batch_size, shuffle=shuffle)

    for i in range(n_epoch):
        start = time.time()
        do_epoch(criterion, model, optimizer, scheduler, train_loader, use_gpu)

        train_loss = validate(model, train_loader, criterion, i, writer, use_gpu)
        writer.add_scalar('Loss/train', train_loss, i+1)
        #val_loss = validate(model, val_loader, criterion, use_gpu)
        end = time.time()
        #history.save(train_loss, val_loss, optimizer.param_groups[0]['lr'])

        print('Epoch {} - Train loss: {:.4f} Training time: {:.2f}s'.format(i, train_loss, end - start))
        #model.save(os.path.join(model_dir, f"model_epoch_{i+1}")
        #print('Epoch {} - Train loss: {:.4f} - Val loss: {:.4f} Training time: {:.2f}s'.format(i,
         #                                                                                      train_loss,
         #                                                                                      val_loss,
          #                                                                                     end - start))
    return history


def do_epoch(criterion, model, optimizer, scheduler, train_loader, use_gpu):
    model.train()
    if scheduler:
        scheduler.step()
    for i, batch in enumerate(train_loader):

        inputs, labels, _ = batch
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        #inputs = Variable(inputs)
        #targets = Variable(targets)
        optimizer.zero_grad()
        output_h = model(inputs)
        loss = criterion(labels, output_h)
        loss.backward()
        optimizer.step()


def predict(model, image_path, folder):
    # TODO adapt for mapping
    model.eval()
    pass
