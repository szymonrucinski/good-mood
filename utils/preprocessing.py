import io
import json
import platform

import librosa.display
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from pydub import AudioSegment

# MacPlot error fix
if platform.system() == "Darwin":
    matplotlib.use("Agg")


def prepare_audio(bytes) -> None:
    """Pre process audio file to be used for prediction"""
    sound_bytes = io.BytesIO(bytes)
    AudioSegment.from_file(sound_bytes).export("sample.wav", format="wav")


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
