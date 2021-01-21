import os
import random
import cv2

from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

from src.player_detection.data.dataset import get_players_dict


# Define hyperparameters
PATH_DATA = 'data/player_detection/'
CLASSES = ['away_player', 'home_player']


def train_player_detection(data_path, classes):

    # classes = ['player']

    for d in ["train", "test", "valid"]:
        DatasetCatalog.register('player_detection_' + d, lambda d=d: get_players_dict(data_path + d + '.csv', data_path + d + '/'))
        MetadataCatalog.get('player_detection_' + d).set(thing_classes=classes)
    player_detection_metadata = MetadataCatalog.get("player_detection_train")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ('player_detection_train',)
    #cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon)
    cfg.MODEL.DEVICE = 'cpu'

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    train_player_detection(PATH_DATA, CLASSES)