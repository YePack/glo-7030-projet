import numpy as np
import torch
import torch.nn as nn
import time
import glob
import os
import math
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path

from src.semantic.dataloader import DataGenerator
from src.semantic.dataloader.dataset import load_image
from src.semantic.dataloader import NormalizeCropTransform
from src.semantic.training_script import NUMBER_OF_CLASSES


def train_valid_loaders(train_path, valid_path, batch_size, transform, shuffle=True):

    # List the files in train and valid
    train_images = glob.glob(str(Path(train_path, '*.png')))
    train_labels = glob.glob(str(Path(train_path, '*.pkl')))
    valid_images = glob.glob(str(Path(valid_path, '*.png')))
    valid_labels = glob.glob(str(Path(valid_path, '*.pkl')))

    train_images.sort()
    train_labels.sort()
    valid_images.sort()
    valid_labels.sort()

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

    train_loader, val_loader = train_valid_loaders(train_path, valid_path, transform=transform,
                                                   batch_size=batch_size, shuffle=shuffle)

    for i in range(n_epoch):
        start = time.time()
        do_epoch(criterion, model, optimizer, scheduler, train_loader, use_gpu, weight_adaptation)

        train_loss = validate(model, train_loader, criterion, use_gpu)

        val_loss = validate(model, val_loader, criterion, use_gpu)
        end = time.time()

        print('Epoch {} - Train loss: {:.4f} - Val loss: {:.4f} Training time: {:.2f}s'.format(i,
                                                                                               train_loss,
                                                                                               val_loss,
                                                                                               end - start))


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
                np.array([1/(np.log(1.1 + (np.array(targets.cpu() == i)).mean())) for i in range(NUMBER_OF_CLASSES)]))
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


def predict(model, image_path, folder):

    # A sortir de la fonction eventuellement pour le normalize ...
    def crop_center(img, cropx, cropy):
        y, x = img.shape[0:2]
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty+cropy, startx:startx+cropx]
    
    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # load the image to predict
    with open(os.path.join(folder, image_path), 'rb') as f:
        image_raw = load_image(f)
        image_predict = np.array(image_raw)[..., :3]
    
    image_predict = crop_center(image_predict, 450, 256)
    img_tensor = transform(image_predict)
    
    img_tensor.unsqueeze_(0)
    img_predicted = model(img_tensor)
    img_predicted = img_predicted.max(dim=1)[1]
    
    return img_predicted, image_raw
