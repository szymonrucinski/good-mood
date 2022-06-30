import librosa.display
from matplotlib import pyplot as plt
import numpy as np
import matplotlib

# MacPlot error fix
matplotlib.use("Agg")


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
