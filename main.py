"""Serve the model as a fastapi app with gradio client"""
import json
import logging
from logging import getLogger
from pathlib import Path

import coloredlogs
import gradio as gr
import librosa
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI
from moviepy.video.io.bindings import mplfig_to_npimage
from PIL import Image

from utils.preprocessing import plot_mel

coloredlogs.install()

logger = getLogger(__name__)


app = FastAPI()
CUSTOM_PATH = "/"


# @app.get("/")
# def read_main():
#     """Return a friendly HTTP greeting."""
#     return {"message": "This is your main app"}


# @app.get("/gradio")
def predict(file_path):
    """Predict the emotion of the audio file"""
    # parse the json data
    logging.info(file_path)
    audio, rate = librosa.load(file_path)
    fig = plot_mel(audio, rate)
    numpy_image = mplfig_to_npimage(fig)

    data = Image.fromarray(numpy_image)
    # data.save("gfg_dummy_pic.jpeg")
    resize = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    input = resize(data).unsqueeze(0)
    f = open(Path(Path().resolve(), "data/responses.json"), encoding="utf-8")
    labels = json.load(f)
    f.close()

    outputs = MODEL.forward(input)
    _, y_hat = outputs.max(1)
    prediction = labels[str(y_hat.item())]
    return prediction


io = gr.Interface(
    fn=predict,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs="text",
)
MODEL = torch.load("model.pt", map_location="cpu")
gradio_app = gr.routes.App.create_app(io)
app.mount(CUSTOM_PATH, gradio_app)
