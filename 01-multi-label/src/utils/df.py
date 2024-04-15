import pandas as pd
import torch
from utils import FileManager
import os


def calculate_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    Calculate the class weights for the dataset.
    :param train_df: The training DataFrame

    Returns:
    class_weights_tensor: The class weights as a tensor
    """ 
    targets = train_df.drop('Image Path', axis=1)
    targets = targets.to_numpy()

    from .handle_class_imbalance import generate_class_weights
    class_weights = generate_class_weights(targets, multi_class=False, one_hot_encoded=True)

    num_labels = len(class_weights)
    values = [class_weights[i] for i in range(num_labels)]
    class_weights_tensor = torch.tensor(values)

    # normalize the class weights
    class_weights_tensor = class_weights_tensor / class_weights_tensor.sum()
    return class_weights_tensor


def get_df(file_manager: FileManager):
    """
    This function will create the DataFrame with the image paths and labels, and split the data into train and validation sets.
    :param file_manager: The file manager object
    :param one_hot: Whether to one-hot encode the labels
    :param multi_class: Whether to remove all columns with more than one class
    :param few_labels: Whether to use a subset of labels

    Returns:
    train_df: The training DataFrame, does not consist of the "No finding" label
    val_df: The validation DataFrame
    test_df: The test DataFrame
    labels: The labels for the diseases, and it consists of the "No finding" label
    class_weights: The class weights
    """

    logger = file_manager.logger
    data_path = file_manager.data_path

    labels = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Effusion",
        "Emphysema",
        "Fibrosis",
        "Hernia",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pleural Thickening",
        "Pneumonia",
        "Pneumothorax"
    ]
    file_path_train = data_path + '/train_official.txt'
    file_path_val = data_path + '/val_official.txt'
    file_path_test = data_path + '/test_official.txt'

    columns = ['Image Filename'] + labels

    df_train = pd.read_csv(file_path_train, sep='\s+', names=columns)
    df_val = pd.read_csv(file_path_val, sep='\s+', names=columns)
    df_test = pd.read_csv(file_path_test, sep='\s+', names=columns)

    # Finding all image paths, and mapping them to the DataFrame
    subfolders = [f"images_{i:03}/images" for i in range(1, 13)]  # Generates 'images_001' to 'images_012'
    path_mapping = {}
    for subfolder in subfolders:
        full_folder_path = os.path.join(data_path, subfolder)
        for img_file in os.listdir(full_folder_path):
            path_mapping[img_file] = os.path.join(full_folder_path, img_file)

    # Update the DataFrame using the mapping
    df_train['Full Image Path'] = df_train['Image Filename'].map(path_mapping)
    df_val['Full Image Path'] = df_val['Image Filename'].map(path_mapping)
    df_test['Full Image Path'] = df_test['Image Filename'].map(path_mapping)

    # Move 'Full Image Path' to the front of the DataFrame
    cols_train = ['Full Image Path'] + [col for col in df_train.columns if col != 'Full Image Path']
    cols_val = ['Full Image Path'] + [col for col in df_val.columns if col != 'Full Image Path']
    cols_test = ['Full Image Path'] + [col for col in df_test.columns if col != 'Full Image Path']
    df_train = df_train[cols_train]
    df_val = df_val[cols_val]
    df_test = df_test[cols_test]

    # Drop 'Image Filename' column
    df_train = df_train.drop(columns=['Image Filename'])
    df_val = df_val.drop(columns=['Image Filename'])
    df_test = df_test.drop(columns=['Image Filename'])


    logger.info(f"Train dataframe shape: {df_train.shape} (1 size larger than expected due to 'Full Image Path')")
    logger.info(f"Train columns: {df_train.columns}")
    logger.info(f"Labels: {labels}")

    # TODO: Add class_weights? :)
    return df_train, df_val, df_test, labels, None
