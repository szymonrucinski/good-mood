import json
import os
import platform

import librosa.display
import matplotlib
import numpy as np
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
