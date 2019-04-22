import torch
import torch.nn as nn
import time

from sklearn.metrics import accuracy_score
from torch.utils.data.sampler import SequentialSampler

from torch.autograd import Variable
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from src.dataloader import DataGenerator


def train_valid_loaders(imagepath_train, labelpath_train, imagepath_val, labelpath_val, batch_size,
                        transform, shuffle=True):

    dataset_train = DataGenerator(imagepath_train, labelpath_train, transform)
    dataset_val = DataGenerator(imagepath_val, labelpath_val, transform)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    return loader_train, loader_val


def validate(model, val_loader, use_gpu=False):
    model.train(False)
    #true = []
    #pred = []
    val_loss = []

    criterion = nn.CrossEntropyLoss()
    model.eval()

    for j, batch in enumerate(val_loader):

        inputs, targets = batch
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)
        output = model(inputs)

        predictions = output.max(dim=1)[1]

        val_loss.append(criterion(output, targets[:, 0]).item())
        #true.extend(targets.data.cpu().numpy().tolist())
        #pred.extend(predictions.data.cpu().numpy().tolist())

    model.train(True)
    #return accuracy_score(true, pred) * 100, sum(val_loss) / len(val_loss)
    return sum(val_loss) / len(val_loss)


def train(model, optimizer, imagepath_train, labelpath_train, imagepath_val, labelpath_val, n_epoch, batch_size,
          transform, use_gpu=False, scheduler=None, criterion=None, shuffle=True):

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    train_loader, val_loader = train_valid_loaders(imagepath_train, labelpath_train, imagepath_val, labelpath_val,
                                                   transform=transform, batch_size=batch_size, shuffle=shuffle)

    for i in range(n_epoch):
        start = time.time()
        do_epoch(criterion, model, optimizer, scheduler, train_loader, use_gpu)
        end = time.time()

        #train_acc, train_loss = validate(model, train_loader, use_gpu)
        train_loss = validate(model, train_loader, use_gpu)
        #val_acc, val_loss = validate(model, val_loader, use_gpu)
        val_loss = validate(model, val_loader, use_gpu)
        #print('Epoch {} - Train acc: {:.2f} - Val acc: {:.2f} - Train loss: {:.4f} - Val loss: {:.4f} - Training time: {:.2f}s'.format(i,
        #                                                                                                      train_acc,
        #                                                                                                      val_acc,
        #                                                                                                      train_loss,
        #                                                                                                      val_loss, end - start))

        print('Epoch {} - Train loss: {:.4f} - Val loss: {:.4f} Training time: {:.2f}s'.format(i,
                                                                                               train_loss,
                                                                                               val_loss,
                                                                                               end - start))


def do_epoch(criterion, model, optimizer, scheduler, train_loader, use_gpu):
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

        loss = criterion(output, targets[:, 0])
        loss.backward()
        optimizer.step()
