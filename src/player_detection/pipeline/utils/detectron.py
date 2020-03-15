import torch

from detectron2.config import get_cfg
from detectron2 import model_zoo


def setup_cfg(config_file,
              weights_file=None,
              config_opts=[],
              confidence_threshold=None,
              cpu=False):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.merge_from_list(config_opts)

    if confidence_threshold is not None:
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold

    if weights_file is not None:
        cfg.MODEL.WEIGHTS = weights_file

    if cpu or not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    cfg.freeze()

    return cfg