
import os
import sys
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torchvision.transforms as transforms
import PIL.Image as Image

def load_and_preprocess_images(image_paths, normalize):
    if normalize.lower() == "imagenet":
        normalize = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
        normalize = transforms.Normalize(
            [0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])

    transform = transforms.Compose([
        # Resize the image to the same size expected by the model
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # Convert the image to a tensor
        normalize
    ])
    images = [transform(Image.open(path).convert('RGB'))
              for path in image_paths]
    batch = torch.stack(images)  # Stack images into a single batch
    
    return batch


def get_bboxes(root_folder):
    images_path = f"{root_folder}/images"
    file_path_bbox = root_folder + '/BBox_List_2017.csv'
    file_path_data_entry = root_folder + '/Data_Entry_2017.csv'

    df_bbox = pd.read_csv(file_path_bbox)
    df_data_entry = pd.read_csv(file_path_data_entry)
    merged_df = pd.merge(df_bbox, df_data_entry, on='Image Index', how='inner')
    merged_df.rename(columns={
        'Bbox [x': 'x',
        'h]': 'h',
    }, inplace=True)
    merged_df['filepath'] = f'{images_path}/' + \
        merged_df['Image Index']
    df = merged_df
    return df


def create_dataloader(data_path, normalization, test_augment, batch_size, num_workers):
    src_dir = os.path.join("/cluster/home/taheeraa/code/BenchmarkTransformers"
                           )
    if src_dir not in sys.path:
        sys.path.append(src_dir)

    from utils import metric_AUROC
    from dataloader import ChestXray14Dataset, build_transform_classification

    images_path = data_path + "/images"
    path_to_labels = '/cluster/home/taheeraa/code/BenchmarkTransformers/dataset'
    file_path_test = path_to_labels + '/Xray14_test_official.txt'

    test_transforms = build_transform_classification(
        normalize=normalization,
        add_augment=True,
        mode="test",
        test_augment=test_augment
    )

    dataset_test = ChestXray14Dataset(images_path=images_path,
                                      file_path=file_path_test,
                                      augment=test_transforms
                                      )

    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)

    return dataloader_test
