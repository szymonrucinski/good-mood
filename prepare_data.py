import os

import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision.transforms as T
from PIL import Image
from skimage import io
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.utils import make_grid

from utils import pytorchtools


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
