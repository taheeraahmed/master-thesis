import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import torch
from utils.plot_stuff import plot_percentage_train_val, plot_number_patient_disease
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch

def calculate_class_weights(integer_labels):
    """
    Calculate class weights given a series of integer labels.
    """
    unique_classes = np.unique(integer_labels)
    class_weights = compute_class_weight(
        'balanced', classes=unique_classes, y=integer_labels)
    return torch.tensor(class_weights, dtype=torch.float)



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


def convert_one_hot_to_integers(df, labels):
    """
    Convert one-hot encoded DataFrame columns for labels into a single column of integer labels.
    """
    labels_df = df[labels]  # Assuming labels are the column names for the one-hot encoded labels
    integer_labels = labels_df.idxmax(axis=1).apply(labels.index)
    return integer_labels


def one_hot_encode(df, labels):
    """
    One-hot encode the diseases in the DataFrame
    :param df: DataFrame with the image paths and labels
    :param diseases: List with the diseases
    """
    for disease in labels:
        df[disease] = df['Finding Labels'].apply(
            lambda x: 1 if disease in x else 0)
    df = df.drop('Finding Labels', axis=1)
    return df

def label_encode(df, labels):
    """
    Label encode the diseases in the DataFrame.
    :param df: DataFrame with the image paths and labels.
    :param labels: List with the diseases.
    """
    # create a mapping from disease to integer
    label2id = {label: i for i, label in enumerate(labels)}
    # extract only the disease columns for idxmax operation
    disease_columns = df[labels]
    # apply the mapping to get the encoded labels
    df['Encoded Labels'] = disease_columns.idxmax(axis=1).map(label2id)
    # keep only the 'Image Path' and 'Encoded Labels' columns in the final DataFrame
    final_df = df[['Image Path', 'Encoded Labels']]

    return final_df

def split_train_val(df, val_size):
    """
    Split the data into train and validation sets

    :param df: DataFrame with the image paths and labels
    :param logger: The  logger object
    """
    patient_ids = df['Patient ID'].unique()
    train_ids, val_ids = train_test_split(
        patient_ids, test_size=val_size, random_state=0)

    train_df = df[df['Patient ID'].isin(train_ids)]
    val_df = df[df['Patient ID'].isin(val_ids)]

    train_df = train_df.drop('Patient ID', axis=1).reset_index(drop=True)
    val_df = val_df.drop('Patient ID', axis=1).reset_index(drop=True)

    return train_df, val_df


def get_labels(df):
    """
    Get the labels from the DataFrame from the column 'Finding Labels' in DataEntry2017.csv,
    excluding 'No Findings'.
    :param df: DataFrame with the image paths and labels
    """
    # explode the 'Finding Labels' into individual labels and get unique values
    labels = df['Finding Labels'].str.split('|').explode().unique()
    # convert to list for easier manipulation
    labels_list = list(labels)
    # sort the list of labels
    labels_list.sort()
    return labels_list


def get_df(file_manager, one_hot=True):
    """
    This function will create the DataFrame with the image paths and labels, and split the data into train and validation sets.
    :param args: The arguments
    :param data_path: The path to the data
    :param logger: The logger object

    Returns:
    train_df: The training DataFrame, does not consist of the "No finding" label
    val_df: The validation DataFrame
    labels: The labels for the diseases, and it consists of the "No finding" label
    class_weights: The class weights
    """

    df = pd.read_csv(f'{file_manager.data_path}/Data_Entry_2017.csv')
    df = get_df_image_paths_labels(df=df, data_path=file_manager.data_path)
    # select the columns we need
    df = df[['Image Path', 'Finding Labels', 'Patient ID']]
    # get the labels from the DataFrame
    labels = get_labels(df)
    # removing all columns with more than one class
    df = df[~df['Finding Labels'].str.contains(r'\|')]
    # one-hot or label encode the diseases
    df = one_hot_encode(df, labels=labels)

    train_df, val_df = split_train_val(
        df=df, val_size=0.2)

    # plot the number of patients with each disease
    plot_number_patient_disease(file_manager=file_manager, df=df, diseases=labels)    

    plot_percentage_train_val(
        file_manager=file_manager,
        train_df=train_df,
        val_df=val_df,
        diseases=labels,
    )

    if one_hot: 
        if 'No Finding' in labels:
            labels.remove('No Finding')
        train_df = train_df.drop('No Finding', axis=1)
        val_df = val_df.drop('No Finding', axis=1)

        integer_labels = convert_one_hot_to_integers(train_df, labels)
        class_weights = calculate_class_weights(integer_labels)
        assert(len(labels) == 14), f"Expected 14 labels, but found {len(labels)}"
    else:
        integer_labels = convert_one_hot_to_integers(train_df, labels)
        train_df = label_encode(train_df, labels=labels)
        val_df = label_encode(val_df, labels=labels)
        class_weights = calculate_class_weights(integer_labels)
        assert(len(labels) == 15), f"Expected 15 labels, but found {len(labels)}"

    file_manager.logger.info(f"Training df\nColumns: {train_df.columns}\nShape: {train_df.shape}")
    file_manager.logger.info(f"Validation df\nColumns: {val_df.columns}\nShape: {val_df.shape}")
    file_manager.logger.info(f"Disease Labels: {labels}")

    return train_df, val_df, labels, class_weights
