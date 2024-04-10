import glob
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from utils import FileManager


def get_df_image_paths_labels(df, data_path):
    """
    This function will add the image paths to the DataFrame
    :param df: DataFrame with the image labels
    :param data_path: Path to the data
    """
    image_paths = []
    for i in range(1, 13):
        folder_name = f'{data_path}/images_{i:03}'
        files_in_subfolder = glob.glob(f'{folder_name}/images/*')
        image_paths.extend(files_in_subfolder)

    assert len(
        image_paths) == 112120, f"Expected 112120 images, but found {len(image_paths)}"
    df['Image Path'] = image_paths
    return df


def one_hot_encode(df, labels):
    """
    One-hot encode the diseases in the DataFrame
    :param df: DataFrame with the image paths and labels
    :param labels: List with the diseases
    """
    for pathology in labels:
        df.loc[:, pathology] = df['Finding Labels'].apply(lambda x: 1 if pathology in x else 0)

    df = df.drop('Finding Labels', axis=1)
    # removing columns with no ground truth labels
    row_label_sums = df.iloc[:, 2:].sum(axis=1)
    df = df[row_label_sums > 0]
    return df

def split_data(df, val_size=0.2, test_size=0.1):
    """
    Split the data into training, validation, and test sets based on unique patient IDs.
    Ensures that all images from the same patient are kept in the same set, which is crucial 
    for medical datasets to prevent information leakage across sets.
    
    :param df: DataFrame containing the image paths, labels, and patient IDs.
    :param val_size: Proportion of the dataset to use for the validation set (after excluding the test set).
    :param test_size: Proportion of the dataset to use for the test set.
    :returns: Three DataFrames corresponding to the training, validation, and test sets.
    """
    # ensure that patient IDs are unique before splitting
    patient_ids = df['Patient ID'].unique()

    # first split: separate out the test set based on patient IDs
    train_val_ids, test_ids = train_test_split(
        patient_ids, test_size=test_size, random_state=42)

    # second split: separate the remaining patient IDs into training and validation sets
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_size / (1 - test_size), random_state=42)

    # use the separated IDs to create the actual dataframes
    train_df = df[df['Patient ID'].isin(train_ids)].reset_index(drop=True)
    val_df = df[df['Patient ID'].isin(val_ids)].reset_index(drop=True)
    test_df = df[df['Patient ID'].isin(test_ids)].reset_index(drop=True)

    # drop the 'Patient ID' column if it's no longer needed
    train_df = train_df.drop('Patient ID', axis=1)
    val_df = val_df.drop('Patient ID', axis=1)
    test_df = test_df.drop('Patient ID', axis=1)

    return train_df, val_df, test_df


def get_labels(df):
    """
    Get the labels from the DataFrame from the column 'Finding Labels' in DataEntry2017.csv,
    excluding 'No Findings'.
    :param df: DataFrame with the image paths and labels
    """
    # explode the 'Finding Labels' into individual labels and get unique values
    labels = df['Finding Labels'].str.split('|').explode().unique()
    labels_list = list(labels) # convert to list for easier manipulation
    labels_list.sort() # sort the list of labels
    return labels_list


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
    root_folder = file_manager.data_path
    labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'No Finding']
    file_path_data_entry = root_folder + '/Data_Entry_2017.csv'

    df = pd.read_csv(file_path_data_entry)
    df = get_df_image_paths_labels(df, root_folder)
    df = df[['Image Path', 'Finding Labels', 'Patient ID']]

    logger.info(f"One-hot encoding labels")
    df = one_hot_encode(df, labels)

    val_size = 0.2
    test_size = 0.1
    train_df, val_df, test_df = split_data(df, val_size=val_size, test_size=test_size)
    logger.info(f"Splitting data into train, validation {val_size}, and test {test_size} sets")

    
    class_weights = calculate_class_weights(train_df)

    logger.info(f"Train dataframe shape: {train_df.shape} (1 size larger than expected due to 'Image Path')")
    logger.info(f"Train columns: {train_df.columns}")
    logger.info(f"Labels: {labels}")
    logger.info(f"Class weights: {class_weights}")
    logger.info(f"Class weights shape: {class_weights.shape}")

    return train_df, val_df, test_df, labels, class_weights
