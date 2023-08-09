import json
import os
import platform
from PIL import Image
import librosa.display
import matplotlib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt

# MacPlot error fix
if platform.system() == "Darwin":
    matplotlib.use("Agg")


def label_encoder_to_json(label_encoder, file_path):
    mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
    file = open(file_path, "w")
    json.dump(mapping, file)
    file.close()


def plot_mel(audio, rate):
    """
    Args:
        audio - vector of audio.
        rate - int sound rate.
    """
    D = np.abs(librosa.stft(audio)) ** 2
    S = librosa.feature.melspectrogram(S=D, sr=rate)
    fig = plt.gcf()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    plt.close()
    return fig


def decompose_emodb(EMODB_PATH):
    "Name Folder same as file extension"
    # EMODB_PATH = './dataset/wav/'
    emotion = []
    path = []
    for root, dirs, files in os.walk(EMODB_PATH):
        for name in files:
            if name[5] == "W":  # Ärger (Wut) -> Angry
                emotion.append("angry")
            elif name[5] == "L":  # Langeweile -> Boredom
                emotion.append("bored")
            elif name[5] == "E":  # Ekel -> Disgusted
                emotion.append("disgust")
            elif name[5] == "A":  # Angst -> Angry
                emotion.append("fear")
            elif name[5] == "F":  # Freude -> Happiness
                emotion.append("happy")
            elif name[5] == "T":  # Trauer -> Sadness
                emotion.append("sad")
            elif name[5] == "N":
                emotion.append("neutral")
            else:
                emotion.append("unknown")
            path.append(os.path.join(EMODB_PATH, name))

    emodb_df = pd.DataFrame(emotion, columns=["labels"])
    emodb_df["source"] = "EMODB"
    emodb_df = pd.concat([emodb_df, pd.DataFrame(path, columns=["path"])], axis=1)

    return emodb_df


def create_audio_spectrogram(
    image_dataset_path: str, audio_dataset_path: str, dataset_summary: pd.DataFrame
) -> None:
    images_count = len(os.listdir(image_dataset_path))
    audio_count = len(os.listdir(audio_dataset_path))

    if images_count != audio_count:
        for path in tqdm(dataset_summary["path"]):
            jpeg_path = (
                path.replace("raw", "preprocessed/images")
                .replace("jpeg/", "")
                .replace("wav", "jpeg")
            )
            print(jpeg_path)
            audio, rate = librosa.load(path)
            fig = plot_mel(audio, rate)
            fig.savefig(jpeg_path)


class EmoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_label = self.df.iloc[idx]["labels"]
        image_path = self.df.iloc[idx]["path"]

        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        return (image, image_label)
