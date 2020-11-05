import torch.nn as nn
import os
import sys
import json
from pathlib import Path

from optparse import OptionParser
import torch
from torch import optim
from src.semantic.model.unet.unet_model import UNet
from src.semantic.training_function import train
from src.semantic.dataloader import NormalizeCropTransform
from src.semantic.loss import DiceCoeff

from src.semantic.utils.show_images_sample import see_image_output
from src.data_creation.file_manager import readfile, savefile


def create_model(model_type, model_params):
    if model_type.lower() == 'vgg16':
        raise NotImplementedError('Need to import the net from model and adapt the script. Old Stuff there')
    if model_type.lower() == 'unet':
        return UNet(**model_params)
    else:
        raise NotImplementedError('Need to specify a valid Neural Network model')


def create_optimizer(optimizer_type, model, optimizer_params):
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if optimizer_type.lower() == "sgd":
        return optim.SGD(trainable_parameters, **optimizer_params)
    else:
        raise NotImplementedError


def create_loss(loss_type):
    if loss_type.lower() == 'crossentropy':
        return nn.CrossEntropyLoss()
    if loss_type.lower() == 'Dice':
        return DiceCoeff()


def create_scheduler():
    return None


def create_device(use_gpu):
    if use_gpu:
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def training(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)

    net = create_model(
        model_type=config["model_type"],
        model_params=config["model_parameters"]
    )
    optimizer = create_optimizer(
        optimizer_type=config["optimizer_type"],
        model=net,
        optimizer_params=config["optimizer_params"]
    )
    loss_criterion = create_loss(config["loss_type"])

    transform = NormalizeCropTransform(**config["transform_params"])

    scheduler = create_scheduler()

    device = create_device(config["use_gpu"])

    net.to(device)

    data_creation_folder_path = config["data_parameters"]["data_creation_folder_path"]
    training_path = Path(data_creation_folder_path, "train")
    validation_path = Path(data_creation_folder_path, "valid")
    testing_path = Path(data_creation_folder_path, "test")

    training_dict = {
        "model": net,
        "optimizer": optimizer,
        "train_path": training_path,
        "valid_path": validation_path,
        "transform": transform,
        "criterion": loss_criterion,
        "device": device,
        "scheduler": scheduler,
        **config["training_parameters"]
    }

    try:
        train(**training_dict)

        see_image_output(
            net,
            path_train=training_path,
            path_test=testing_path,
            path_save=data_creation_folder_path
        )

    except KeyboardInterrupt:
        savefile(net, config["model_save_name"])

        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    net.to(torch.device("cpu"))
    savefile(net, config["model_save_name"])


def get_args():
    parser = OptionParser()
    parser.add_option('-c', '--config', type=str, dest='config',
                      help='Config file to setup training')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    training(args.config)
