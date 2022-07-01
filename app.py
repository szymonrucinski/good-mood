from email.mime import audio
from email.policy import strict
from multiprocessing import process
from tabnanny import process_tokens
import flask
import io
import string
import time
import os
import librosa
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np
import soundfile as sf
from utils.preprocessing import plot_mel
import torchvision.transforms as transforms
import torch
from scipy.io.wavfile import write, read
from PIL import Image
from flask import Flask, jsonify, request
from pydub import AudioSegment
import json
import subprocess


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

model = torch.load("model.pt")
f = open("labels.json")
labels = json.load(f)
f.close()

### PRINT SCRIPT OUTPUT
def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


# prepare audio
def prepare_audio(bytes):
    s = io.BytesIO(bytes)
    AudioSegment.from_file(s).export("sample.wav", format="wav")


app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return "Machine Learning Inference"


@app.route("/train", methods=["GET"])
def train_api():

    command = "conda run -n ml jupyter nbconvert Report_SER.ipynb --to script && conda run -n ml python3 Report_SER.py"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    command = "conda run -n ml python3 Report_SER.py"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    process.wait()

    return "Training started"


# Example


@app.route("/predict", methods=["POST"])
def predict_api():
    # Catch the image file from a POST request
    print(request)

    if "file" not in request.files:
        return "Please try again. The Audio doesn't exist"
    print(request)
    file = request.files.get("file")
    if not file:
        return
    # Read the image
    bytes = file.read()
    prepare_audio(bytes)
    audio, rate = librosa.load("sample.wav")
    fig = plot_mel(audio, rate)
    numpy_image = mplfig_to_npimage(fig)

    data = Image.fromarray(numpy_image)
    data.save("gfg_dummy_pic.jpeg")

    model = torch.load("model.pt", map_location="cpu")
    resize = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    input = resize(data).unsqueeze(0)

    outputs = model.forward(input)
    _, y_hat = outputs.max(1)
    prediction = labels[str(y_hat.item())]
    # Return on a JSON format
    return prediction


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 4444)))
