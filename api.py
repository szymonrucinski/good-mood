from email.mime import audio
import flask
import io
import string
import time
import os
import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.io.wavfile import write, read
from PIL import Image
from flask import Flask, jsonify, request
from pydub import AudioSegment

# def convert_bytearray_to_wav_ndarray(input_bytearray: bytes, sampling_rate=16000):
#     bytes_wav = bytes()
#     byte_io = io.BytesIO(bytes_wav)
#     write(byte_io, sampling_rate, np.frombuffer(input_bytearray, dtype=np.int16))
#     output_wav = byte_io.read()
#     scipy.io.wavfile.read()
#     wav_r = np.fromstring(wav_bytes, dtype=np.uint8)

#     return output



#prepare audio
def prepare_audio(bytes):
    # with open('sample.wav', mode='bx') as f:
    #     f.write(bytes)
    s = io.BytesIO(bytes)
    AudioSegment.from_file(s).export('Tomek.wav', format='wav')
    audio = AudioSegment.from_file(s)

    return audio

       # audio, rate = sf.read(io.BytesIO(bytes))
    # return audio, rate

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


@app.route('/train', methods=['GET'])
def train_api():
    # Return on a JSON format
    return 'Training started'

@app.route('/predict', methods=['POST'])
def predict_api():
    #Catch the image file from a POST request

    print(request)

    if 'file' not in request.files:
        return "Please try again. The Audio doesn't exist"
    print(request)
    file = request.files.get('file')
    if not file:
        return

    # Read the image
    bytes = file.read()

    # Prepare the image
    audio = prepare_audio(bytes)

    
    # Return on a JSON format
    return 'rate'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=int(os.getenv('PORT', 4444)))