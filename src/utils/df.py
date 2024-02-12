import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import torch
from utils.plot_stuff import plot_percentage_train_val, plot_number_patient_disease

def get_df_image_paths_labels(args, data_path, logger):
    df = pd.read_csv(f'{data_path}/Data_Entry_2017.csv')
    image_paths = []
    for i in range(1, 13):
        folder_name = f'{data_path}/images_{i:03}'
        files_in_subfolder = glob.glob(f'{folder_name}/images/*')
        image_paths.extend(files_in_subfolder)

    assert len(image_paths) == 112120, f"Expected 112120 images, but found {len(image_paths)}"
    df['Image Path'] = image_paths
    df = df[['Image Path', 'Finding Labels', 'Patient ID']]

    diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
                'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
                'Pneumonia', 'Pneumothorax']
    
    # one-hot encoding disease and dropping finding labels
    for disease in diseases:
        df[disease] = df['Finding Labels'].apply(lambda x: 1 if disease in x else 0)
    df = df.drop('Finding Labels', axis=1)

    plot_number_patient_disease(df, diseases, image_output = f'output/{args.output_folder}/images/number_patient_disease.png')

    # used for handling data leak
    patient_ids = df['Patient ID'].unique()

    # split the patient IDs into train, validation, and test sets
    train_ids, val_ids = train_test_split(patient_ids, test_size=0.2, random_state=0)

    train_df = df[df['Patient ID'].isin(train_ids)]
    val_df = df[df['Patient ID'].isin(val_ids)]

    # check the shapes of the dataframes
    logger.info(f'train_df shape: {train_df.shape}, val_df shape: {val_df.shape}')
    # check the ratios of the dataframes as we split based on patient IDs, not individual images
    logger.info(f'train_df ratio: {round(len(train_df) / len(df), 3)}, val_df ratio: {round(len(val_df) / len(df), 3)}')

    # drop the 'Patient ID' column
    train_df = train_df.drop('Patient ID', axis=1).reset_index(drop=True)
    val_df = val_df.drop('Patient ID', axis=1).reset_index(drop=True)

    # calculate the percentages of each disease in the train, validation, and test sets
    plot_percentage_train_val(train_df = train_df,
                                   val_df = val_df,
                                   diseases = diseases,
                                   image_output = f'output/{args.output_folder}/images/percentage_class_train_val_test.png'
                                   )
    return train_df, val_df 