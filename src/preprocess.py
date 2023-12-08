import os
from torchaudio import transforms as T
import torchaudio
import librosa as lb
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms import v2 as vision_transforms
from src.utils import *
from src.configuration import *
import skimage
import torchvision


def are_features_extracted(save_path: str) -> bool:
    """Checks if the features have already been extracted and saved

    Args:
        save_path (str): Path to the directory where the features are saved

    Returns:
        bool: True if the features are present, False otherwise
    """
    if os.path.exists(save_path):
        for genre in os.listdir(save_path):
            genre_path = os.path.join(save_path, genre)
            for song in os.listdir(genre_path):
                song_path = os.path.join(genre_path, song)
                if not os.path.exists(song_path):
                    return False
        return True
    return False

def generate_spectrograms(data_path: str, save_path: str):
    """Generates and stores waveform, spectrograms, mel spectrograms and mfcc features for each song in the dataset

    Args:
        data_path (str): Original dataset path
        save_path (str): Path to save the extracted features
    """
    transforms = vision_transforms.Compose([
            vision_transforms.Pad(10, padding_mode="constant", fill=255),
            vision_transforms.Resize((432, 288), antialias=True),
    ])
    # Iterate over each genre
    for genre in tqdm(os.listdir(data_path), desc="Genres Processed", dynamic_ncols=True):
        genre_path = os.path.join(data_path, genre)
        
        # Create a directory for each genre in the save_path
        genre_save_path = os.path.join(save_path, genre)
        os.makedirs(genre_save_path, exist_ok=True)

        # Iterate over each song in the genre
        for song in tqdm(os.listdir(genre_path), desc=f"Songs ({genre}) Processed", dynamic_ncols=True):
            song_path = os.path.join(genre_path, song)
            
            # Create a unique folder for each song within the genre
            song_save_folder = os.path.splitext(song)[0]
            song_save_path = os.path.join(genre_save_path, song_save_folder)
            os.makedirs(song_save_path, exist_ok=True)
            
            waveform, sample_rate = torchaudio.load(song_path)
            
            # Compute spectrogram, mel spectrogram and mfcc features
            specgram = T.AmplitudeToDB()(T.Spectrogram()(waveform)[0])
            melspecgram = T.AmplitudeToDB()(T.MelSpectrogram(sample_rate=sample_rate, n_mels=64)(waveform)[0])
            mfcc = T.AmplitudeToDB()(T.MFCC(sample_rate=sample_rate)(waveform)[0])
            features = {
                "waveform": waveform,
                "spectrogram": specgram,
                "melspectrogram": melspecgram,
                "mfcc": mfcc
            }
            # Iterate over each feature extracted
            for feature_name in features:
                feature_save_path = os.path.join(song_save_path, feature_name)
                # If the feature is the waveform, save it as a raw waveform tensor
                if feature_name == "waveform":
                    torch.save(features[feature_name], feature_save_path + TENSOR_EXTENSION)
                # Otherwise, save it as an image
                else:
                    # Save the extracted feature as an image
                    plt.imshow(features[feature_name], origin="lower", cmap="viridis", interpolation="nearest", aspect="auto")
                    plt.axis('off')
                    plt.savefig(feature_save_path + IMG_EXTENSION, bbox_inches='tight', pad_inches=0.1)
                    plt.close()