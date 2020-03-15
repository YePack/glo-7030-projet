import os
import pandas
import numpy
import cv2
import itertools

from PIL import Image
from torch.utils.data import Dataset
from detectron2.structures import BoxMode


def get_players_dict(csv_file,
                     img_dir,
                     classes_file='src/player_detection/data/classes.csv'):
    df = pandas.read_csv(csv_file)
    df_classes = pandas.read_csv(classes_file)

    classes = df_classes['label'].tolist()

    df['filename'] = df['image_id'].map(lambda x: img_dir + x)
    df['label_int'] = df['label'].map(lambda x: classes.index(x))

    dataset_dicts = []
    for filename in df['filename'].unique().tolist():
        record = {}

        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["height"] = height
        record["width"] = width

        objs = []
        for index, row in df[(df['filename'] == filename)].iterrows():
            obj = {
                'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': row['label_int']
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return (dataset_dicts)


class DataBoxesGenerator(Dataset):
    def __init__(self, images_path, annotations_path, transform):
        self.transform = transform
        self.images = images_path
        # Load annotations file
        dtype = {
            'image_filename': 'str',
            'classname': 'str',
            'x0': 'int',
            'y0': 'int',
            'x1': 'int',
            'y1': 'int',
            'class': 'int'
        }
        self.annotations = pandas.read_csv(annotations_path, dtype=dtype)

    def __getitem__(self, index):
        # Load image
        selected_image = self.images[index]
        with open(selected_image, 'rb') as f:
            image = numpy.array(Image.open(f))[..., :3]

        # Find boxes
        selected_annotations = self.annotations.loc[
            self.annotations['image_filename'] == selected_image.split(
                '/')[-1]]
        boxes_array = numpy.zeros((len(selected_annotations), 5))

        for idx, annot in selected_annotations.iterrows():
            boxes_array[idx, 0] = float(annot['x0'])
            boxes_array[idx, 1] = float(annot['y0'])
            boxes_array[idx, 2] = float(annot['x1'])
            boxes_array[idx, 3] = float(annot['y1'])
            boxes_array[idx, 4] = annot['class']

        img_tensor, boxes_tensor = self.transform.fit(image, boxes_array)

        return img_tensor, boxes_tensor

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    get_players_dict('data/player_detection/train.csv',
                     'data/player_detection/train/')
