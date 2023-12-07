import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from src.configuration import *

class GTZANFeatureDataset(Dataset):
    """Dataset-extending class for features extracted from GTZAN dataset."""
    def __init__(self, feature_path: str = "datasets/features", split: str = "train", selected_features = ["waveform", "spectrogram", "melspectrogram", "mfcc"]):
        # split can be "train", "valid" or "test"
        self.feature_path = feature_path
        self.split = split
        self.selected_features = selected_features
        self.genres = CLASS_LABELS
        if not os.path.isdir(self.feature_path):
            raise RuntimeError("Features not found. Please recheck the folder.")
        split_instances_list = os.path.join(self.feature_path, f"../{self.split}_filtered.txt")
        with open(split_instances_list) as f:
            lines = f.readlines()
        self.song_folder_list = [line.strip() for line in lines]

    def __getitem__(self, index):
        song_folder = self.song_list[index]
        genre_idx = CLASS_LABELS.index(song_folder.split("/")[0])
        song_folder_path = os.path.join(self.feature_path, song_folder)
        waveform, spectrogram, melspectrogram, mfcc = None, None, None, None
        if "waveform" in self.selected_features:
            waveform = torch.load(os.path.join(song_folder_path, "waveform"+TENSOR_EXTENSION))
        if "spectrogram" in self.selected_features:
            spectrogram = read_image(os.path.join(song_folder_path, "spectrogram"+IMG_EXTENSION))
        if "melspectrogram" in self.selected_features:
            melspectrogram = read_image(os.path.join(song_folder_path, "melspectrogram"+IMG_EXTENSION))
        if "mfcc" in self.selected_features:
            mfcc = read_image(os.path.join(song_folder_path, "mfcc"+IMG_EXTENSION))
        return waveform, spectrogram, melspectrogram, mfcc, genre_idx

    def __len__(self):
        return len(self.song_folder_list)

def get_ensemble_dataloader(split: str = "train", batch_size = 32, selected_features = ["waveform", "spectrogram", "melspectrogram", "mfcc"]):
    is_shuffle = True if (split == "train") else False
    data_loader = DataLoader(dataset=GTZANFeatureDataset(split=split, selected_features=selected_features), batch_size=batch_size, shuffle=is_shuffle)
    return data_loader

def get_waveform_dataloader(split: str = "train", batch_size = 32):
    selected_features = ["waveform"]
    get_ensemble_dataloader(split=split, batch_size=batch_size, selected_features=selected_features)
    
def get_spectrogram_dataloader(split: str = "train", batch_size = 32):
    selected_features = ["spectrogram"]
    get_ensemble_dataloader(split=split, batch_size=batch_size, selected_features=selected_features)
    
def get_melspectrogram_dataloader(split: str = "train", batch_size = 32):
    selected_features = ["melspectrogram"]
    get_ensemble_dataloader(split=split, batch_size=batch_size, selected_features=selected_features)
    
def get_mfcc_dataloader(split: str = "train", batch_size = 32):
    selected_features = ["mfcc"]
    get_ensemble_dataloader(split=split, batch_size=batch_size, selected_features=selected_features)