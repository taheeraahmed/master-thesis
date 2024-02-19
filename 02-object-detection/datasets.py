from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os


class ChestXRay14BBoxDatast(Dataset):
    def __init__(self, root_folder, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_folder (string): Dictionary with all the paths to a certain x-ray.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_folder = root_folder
        self.bbox_frame = pd.read_csv(root_folder + 'BBox_List_2017.csv')

        # creates an index with all the file names (have to do this in order to get the full path to the images
        # because BBox_List_2017.csv only has the file names)
        self.img_paths = self._create_file_index(root_folder)
        self.transform = transform

    def __len__(self):
        return len(self.bbox_frame)

    def __getitem__(self, idx):
        # find img name in self.bb_frame dataframe given the idx
        img_name = self.bbox_frame.iloc[idx, 0]
        # using self.imgs_paths dictionary to find the full path to the image
        img_path = self.img_paths[img_name]
        # opening the image
        img = Image.open(img_path)

        # getting the bbox
        bbox = self.bbox_frame.iloc[idx, 2:6].values
        bbox = bbox.astype('float').reshape(-1, 4)

        if self.transform:
            img = self.transform(img)

        return {'img': img, 'bbox': bbox}

    def _create_file_index(self, root_folder):
        """
        Creates a dictionary that maps filenames to their full paths within the root_folder.
        """
        file_index = {}
        for root, _, files in os.walk(root_folder):
            for file in files:
                file_index[file] = os.path.join(root, file)
        return file_index
