import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2 as T
from src.configuration import *

class GTZANFeatureDataset(Dataset):
    """Dataset-extending class for features extracted from GTZAN dataset."""
    def __init__(self, feature_path: str = "datasets/features", split: str = "train", selected_feature: str = "melspectrogram"):
        # split can be "train", "valid" or "test"
        self.feature_path = feature_path
        self.split = split
        self.selected_feature = selected_feature
        self.genres = CLASS_LABELS
        if not os.path.isdir(self.feature_path):
            raise RuntimeError("Features not found. Please recheck the folder.")
        split_instances_list = os.path.join(self.feature_path, f"../{self.split}_filtered.txt")
        with open(split_instances_list) as f:
            lines = f.readlines()
        self.song_folder_list = [line.strip() for line in lines]

    def __getitem__(self, index):
        song_folder = self.song_folder_list[index]
        genre_idx = CLASS_LABELS.index(song_folder.split("/")[0])
        song_folder_path = os.path.join(self.feature_path, song_folder)
        feature = None
        if "waveform" == self.selected_feature:
            feature = torch.load(os.path.join(song_folder_path, "waveform"+TENSOR_EXTENSION)).squeeze()
        if "spectrogram" == self.selected_feature:
            feature = read_image(os.path.join(song_folder_path, "spectrogram"+IMG_EXTENSION), mode=ImageReadMode.RGB).squeeze()
        if "melspectrogram" in self.selected_feature:
            feature = read_image(os.path.join(song_folder_path, "melspectrogram"+IMG_EXTENSION), mode=ImageReadMode.RGB).squeeze()
        if "mfcc" in self.selected_feature:
            feature = read_image(os.path.join(song_folder_path, "mfcc"+IMG_EXTENSION), mode=ImageReadMode.RGB).squeeze()
        return feature, genre_idx

    def __len__(self):
        return len(self.song_folder_list)

def get_dataloader(split: str = "train", batch_size = 32, selected_feature = "melspectrogram"):
    is_shuffle = True if (split == "train") else False
    data_loader = DataLoader(dataset=GTZANFeatureDataset(split=split, selected_feature=selected_feature), batch_size=batch_size, shuffle=is_shuffle)
    return data_loader