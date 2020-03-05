p_weight_augmentation = None
p_bilinear = True
p_model_name_save = 'unet'
p_normalize = True
p_max_images= 2000
p_number_of_classes = 9
p_label_to_int = {'ice': 1, 'board': 2, 'circlezone': 3, 'circlemid': 4, 'goal': 5, 'blue': 6, 'red': 7, 'fo': 8}
p_classes_color = ['black', 'white', 'yellow', 'pink', 'coral', 'crimson', 'blue', 'red', 'magenta']
p_history_save_name = None
p_save_name = 'history'

net_dict = {
    'p_weight_augmentation': p_weight_augmentation,
    'p_bilinear': p_bilinear,
    'p_model_name_save': p_model_name_save,
    'p_normalize':  p_normalize,
    'p_max_images': p_max_images,
    'p_number_of_classes': p_number_of_classes,
    'p_label_to_int': p_label_to_int,
    'p_classes_color': p_classes_color
}