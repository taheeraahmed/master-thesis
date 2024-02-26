import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import torch
from utils.plot_stuff import plot_percentage_train_val, plot_number_patient_disease


def get_one_image_per_class(df):
    """
    This function will make sure that there is at least one class or zero classes for each image in the DataFrame.
    If there are more than two classes in an image it will be removed from the dataframe.
    :param df: DataFrame with the validation set
    """
    df['Finding_Sum'] = df.iloc[:, 3:].sum(axis=1)
    df = df[df['Finding_Sum'] == 1 & df['Finding_Sum'] == 0]
    df = df.drop('Finding_Sum', axis=1)
    return df


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


def one_hot_encode(df, diseases):
    """
    One-hot encode the diseases in the DataFrame
    :param df: DataFrame with the image paths and labels
    :param diseases: List with the diseases
    """
    for disease in diseases:
        df[disease] = df['Finding Labels'].apply(
            lambda x: 1 if disease in x else 0)
    df = df.drop('Finding Labels', axis=1)
    return df


def split_train_val(df, logger):
    """
    Split the data into train and validation sets
    
    :param df: DataFrame with the image paths and labels
    :param logger: The  logger object
    """
    patient_ids = df['Patient ID'].unique()
    train_ids, val_ids = train_test_split(
        patient_ids, test_size=0.2, random_state=0)

    train_df = df[df['Patient ID'].isin(train_ids)]
    val_df = df[df['Patient ID'].isin(val_ids)]

    logger.info(
        f'train_df shape: {train_df.shape}, val_df shape: {val_df.shape}')
    logger.info(
        f'train_df ratio: {round(len(train_df) / len(df), 3)}, val_df ratio: {round(len(val_df) / len(df), 3)}')

    train_df = train_df.drop('Patient ID', axis=1).reset_index(drop=True)
    val_df = val_df.drop('Patient ID', axis=1).reset_index(drop=True)

    return train_df, val_df


def get_df(args, data_path, logger):
    """
    This function will create the DataFrame with the image paths and labels, and split the data into train and validation sets.
    :param args: The arguments
    :param data_path: The path to the data
    :param logger: The logger object
    """
    diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
                'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
                'Pneumonia', 'Pneumothorax']
    df = pd.read_csv(f'{data_path}/Data_Entry_2017.csv')
    df = get_df_image_paths_labels(df=df, data_path=data_path)
    df = df[['Image Path', 'Finding Labels', 'Patient ID']]

    # one-hot encoding disease and dropping finding labels
    df = one_hot_encode(df, diseases)

    # remove images with more than one disease
    df = get_one_image_per_class(df)

    # split the data into train and validation sets
    train_df, val_df = split_train_val(
        df, train_size=0.8, val_size=0.2, logger=logger)

    # plot the number of patients with each disease
    try:
        plot_number_patient_disease(
            df, diseases, image_output=f'output/{args.output_folder}/images/number_patient_disease.png')
    except Exception as e:
        logger.error(f'Error plotting number_patient_disease: {e}')

    # calculate the percentages of each disease in the train, validation, and test sets
    try:
        plot_percentage_train_val(train_df=train_df,
                                  val_df=val_df,
                                  diseases=diseases,
                                  image_output=f'output/{args.output_folder}/images/percentage_class_train_val_test.png'
                                  )
    except Exception as e:
        logger.error(f'Error plotting percentage_train_val: {e}')

    return train_df, val_df
