import os
import argparse
import random
import cv2
import matplotlib.pyplot as plt

from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

from src.player_detection.data.dataset import get_players_dict

CLASSES = ['a', 'h']


if __name__ == "__main__":

    for d in ["test"]:
        DatasetCatalog.register('player_detection_' + d, lambda d=d: get_players_dict(data_path + d + '.csv', data_path + d + '/'))
        MetadataCatalog.get('player_detection_' + d).set(thing_classes=CLASSES)
    player_detection_metadata = MetadataCatalog.get("player_detection_test")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ('player_detection_train',)
    cfg.DATASETS.TEST = ('player_detection_test', )
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon)
    cfg.MODEL.WEIGHTS = "models/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_players_dict('data/player_detection/test.csv', 'data/player_detection/test/')
    i = 0
    for d in random.sample(dataset_dicts, 5):
        i = i + 1
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=player_detection_metadata, scale=0.8)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(f'test{i}.png', v.get_image()[:, :, ::-1])