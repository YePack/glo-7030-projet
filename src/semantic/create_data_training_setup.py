import json
from pathlib import Path
from optparse import OptionParser

from src.semantic.modeling_data_creation.split_modeling_data import create_labels_from_dir
from src.semantic.dataloader.flip_images import flip_images


def get_args():
    parser = OptionParser()
    parser.add_option('-c', '--config', type=str, dest='config',
                      help='Config file to setup training')

    (options, args) = parser.parse_args()
    return options


def data_creation(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
    data_parameters = config["data_parameters"]
    # Split train and test in 3 different folders (and save arrays instead of XMLs)
    create_labels_from_dir(
        path_data=data_parameters["raw_data_path"],
        path_to=data_parameters["data_creation_folder_path"],
        train_test_perc=data_parameters["train_test_perc"],
        train_valid_perc=data_parameters["train_valid_perc"],
        max=data_parameters["max_image"]
    )

    if data_parameters["data_augmentation"]:
        train_data_path = Path(data_parameters["data_creation_folder_path"], "train")
        flip_images(train_data_path)


if __name__ == "__main__":
    args = get_args()
    data_creation(args.config)
