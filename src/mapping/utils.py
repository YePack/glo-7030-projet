import os
from pathlib import PurePath

from src.semantic.unet.generate_masks import create_labels_from_dir
from src.semantic.training_function import predict
from src.semantic.utils.utils import savefile


def create_mapping_data(semantic_model, path_data, path_to, train_test_perc=0.8, train_valid_perc=0.8, shuffle=True, max=None):
    create_labels_from_dir(path_data, path_to, train_test_perc, train_valid_perc, shuffle, max)
    split_folds = ['train']  # , 'valid', 'test']  # Just train for now
    for split_fold in split_folds:
        folder_fold = os.path.join(path_to, split_fold)
        files_predict = [f for f in os.listdir(folder_fold) if f.endswith(".png")]
        files_predict.sort()
        for file_predict in files_predict:
            output_sem, _ = predict(semantic_model, file_predict, folder_fold, after_argmax=False)
            semantic_name = PurePath(file_predict).stem + '_semantic' + PurePath(file_predict).suffix
            savefile(output_sem, os.path.join(folder_fold, semantic_name))
