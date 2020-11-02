import torch.nn as nn
import os
import sys
import json
from pathlib import Path

from optparse import OptionParser
from torch import optim
from src.semantic.unet.unet_model import UNet
from src.semantic import train
from src.semantic.dataloader import NormalizeCropTransform
from src.semantic.loss import DiceCoeff

from src.semantic.utils.show_images_sample import see_image_output
from src.data_creation.file_manager import readfile, savefile
from src.semantic.vgg.vggnet import vgg16_bn


NUMBER_OF_CLASSES = 9


def create_model(model_type, model_params):
    if model_type.lower() == 'vgg16':
        return vgg16_bn(**model_params)
    if model_type.lower() == 'unet':
        return UNet(**model_params)
    else:
        raise NotImplementedError('Need to specify a valid Neural Network model')


def create_optimizer(optimizer_type, model, optimizer_params):
    trainable_parameters = [p for p in model.parameters() if p.required_grad]
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

    if config["use_gpu"]:
        net.cuda()

    training_path = Path(config["data_creation_folder_path"], "train")
    validation_path = Path(config["data_creation_folder_path"], "valid")
    testing_path = Path(config["data_creation_folder_path"], "test")

    training_dict = {
        "model": net,
        "optimizer": optimizer,
        "train_path": training_path,
        "valid_path": validation_path,
        "transform": transform,
        "criterion": loss_criterion,
        "use_gpu": config["use_gpu"],
        "scheduler": scheduler,
        **config["training_parameters"]

    }

    try:
        train(**training_dict)

        see_image_output(
            net,
            path_train=training_path,
            path_test=testing_path,
            path_save=config["data_creation_folder_path"]
        )

    except KeyboardInterrupt:
        savefile(net, config["model_save_name"])

        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


    net.cpu()
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




