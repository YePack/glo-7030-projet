import cv2
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer

from src.player_detection.pipeline.pipeline import Pipeline
from src.player_detection.data.dataset import get_players_dict


class AnnotateImage(Pipeline):
    """Pipeline task for image annotation."""

    def __init__(self, dst, classes, instance_mode=ColorMode.IMAGE):
        self.dst = dst

        DatasetCatalog.register('player_detection_train', lambda: get_players_dict('data/player_detection/train.csv', 'data/player_detection/train/'))
        self.metadata = MetadataCatalog.get('player_detection_train').set(thing_classes=classes)
        self.instance_mode = instance_mode
        self.cpu_device = torch.device("cpu")

        super().__init__()

    def map(self, data):
        dst_image = data["image"].copy()
        data[self.dst] = dst_image

        self.annotate_predictions(data)

        return data

    def annotate_predictions(self, data):
        if "predictions" not in data:
            return

        predictions = data["predictions"]
        dst_image = data[self.dst]
        dst_image = dst_image[:, :, ::-1]  # Convert OpenCV BGR to RGB format

        visualizer = Visualizer(dst_image, self.metadata, instance_mode=self.instance_mode)

        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_image = visualizer.draw_panoptic_seg_predictions(panoptic_seg.to(self.cpu_device),
                                                                 segments_info)
        elif "sem_seg" in predictions:
            sem_seg = predictions["sem_seg"].argmax(dim=0)
            vis_image = visualizer.draw_sem_seg(sem_seg.to(self.cpu_device))
        elif "instances" in predictions:
            instances = predictions["instances"]
            vis_image = visualizer.draw_instance_predictions(instances.to(self.cpu_device))

        # Converts RGB format to OpenCV BGR format
        vis_image = cv2.cvtColor(vis_image.get_image(), cv2.COLOR_RGB2BGR)
        data[self.dst] = vis_image