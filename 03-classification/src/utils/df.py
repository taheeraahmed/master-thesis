import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import torch
from utils.plot_stuff import plot_percentage_train_val, plot_number_patient_disease


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
    # Create a mapping from disease to integer
    label_to_int = {disease: i for i, disease in enumerate(labels)}
    # Apply the mapping to the 'Finding Labels' column

    df['Encoded Labels'] = df['Finding Labels'].apply(
        lambda x: label_to_int[x])

    # Drop the original 'Finding Labels' column
    df = df.drop('Finding Labels', axis=1)

    return df


def split_train_val(df, val_size, logger):
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

    logger.info(
        f'train_df shape: {train_df.shape}, val_df shape: {val_df.shape}')
    logger.info(
        f'train_df ratio: {round(len(train_df) / len(df), 3)}, val_df ratio: {round(len(val_df) / len(df), 3)}')

    train_df = train_df.drop('Patient ID', axis=1).reset_index(drop=True)
    val_df = val_df.drop('Patient ID', axis=1).reset_index(drop=True)

    return train_df, val_df


def get_labels(df):
    """
    Get the labels from the DataFrame from the column 'Finding Labels' in DataEntry2017.csv,
    excluding 'No Findings'.
    :param df: DataFrame with the image paths and labels
    """
    # Explode the 'Finding Labels' into individual labels and get unique values
    labels = df['Finding Labels'].str.split('|').explode().unique()
    
    # Convert to list for easier manipulation
    labels_list = list(labels)
    
    # Remove 'No Findings' from the list, if it exists
    if 'No Findings' in labels_list:
        labels_list.remove('No Findings')
    
    # Sort the list of labels
    labels_list.sort()
    
    return labels_list


def get_df(args, data_path, logger):
    """
    This function will create the DataFrame with the image paths and labels, and split the data into train and validation sets.
    :param args: The arguments
    :param data_path: The path to the data
    :param logger: The logger object
    """

    df = pd.read_csv(f'{data_path}/Data_Entry_2017.csv')
    df = get_df_image_paths_labels(df=df, data_path=data_path)
    # select the columns we need
    df = df[['Image Path', 'Finding Labels', 'Patient ID']]
    # get the labels from the DataFrame
    labels = get_labels(df)
    # removing all columns with more than one class
    df = df[~df['Finding Labels'].str.contains(r'\|')]
    # one-hot or label encode the diseases
    df = label_encode(df, labels=labels)
    # df = one_hot_encode(df, labels=labels)

    train_df, val_df = split_train_val(
        df=df, val_size=0.2, logger=logger)

    # plot the number of patients with each disease
    try:
        plot_number_patient_disease(
            df, labels, image_output=f'output/{args.output_folder}/images/number_patient_disease.png')
    except Exception as e:
        logger.error(f'Error plotting number_patient_disease: {e}')

    # calculate the percentages of each disease in the train, validation, and test sets
    try:
        plot_percentage_train_val(train_df=train_df,
                                  val_df=val_df,
                                  diseases=labels,
                                  image_output=f'output/{args.output_folder}/images/percentage_class_train_val_test.png'
                                  )
    except Exception as e:
        logger.error(f'Error plotting percentage_train_val: {e}')

    return train_df, val_df, labels
