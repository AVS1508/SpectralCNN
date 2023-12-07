import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import librosa as lb
import numpy as np

from src.configuration import *

def print_stats(audio_data):
    waveform, sample_rate, label = audio_data
    print("-" * 10)
    print("Label:", label)
    print("-" * 10)
    print("Sample Rate:", sample_rate)
    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    print(f" - Max:     {waveform.max().item():6.3e}")
    print(f" - Min:     {waveform.min().item():6.3e}")
    print(f" - Mean:    {waveform.mean().item():6.3e}")
    print(f" - Std Dev: {waveform.std().item():6.3e}")
    print(waveform)

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def plot_waveform(audio_data, title=None, ax=None):
    waveform, sample_rate, label = audio_data
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    if title is not None:
        ax.set_title(title)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])


def plot_spectrogram(audio_data, type="spectrogram", title=None, ax=None):
    waveform, sample_rate, label = audio_data
    if type == "melspectrogram":
        specgram = T.MelSpectrogram(sample_rate=sample_rate, n_mels=64)(waveform)[0]
    elif type == "mfcc":
        specgram = T.MFCC(sample_rate=sample_rate)(waveform)[0]
    else:
        specgram = T.Spectrogram()(waveform)[0]
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel("Frequency Bins" if type != "melspectrogram" else "Mel Frequency")
    ax.imshow(
        lb.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest"
    )


def random_feature_display(audio_dataset: Dataset, feature: str, suptitle: str, num_display_per_class: int = 5):
    fig, axs = plt.subplots(
        nrows=len(CLASS_LABELS),
        ncols=num_display_per_class,
        figsize=(32 / 5 * num_display_per_class, 32),
    )
    for cdx, class_label in enumerate(CLASS_LABELS):
        for idx in range(num_display_per_class):
            selected_ax = axs[cdx, idx] if num_display_per_class != 1 else axs[cdx]
            select_idx = np.random.randint(0, 100)
            if feature == "waveplot":
                plot_waveform(audio_dataset[cdx * 100 + select_idx], ax=selected_ax)
            else:
                plot_spectrogram(
                    audio_dataset[cdx * 100 + select_idx], type=feature, ax=selected_ax
                )
            selected_ax.set_title(f"{class_label}.{str(select_idx).zfill(5)}.wav")
    fig.suptitle(suptitle, y=1.0)
    fig.tight_layout()
    plt.show()
