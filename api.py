import flask
import io
import string
import time
import os
import numpy as np
import torch
from PIL import Image
from flask import Flask, jsonify, request 

app = Flask(__name__)

def train_model():
    print('I am starting a training')
    pass

@app.route('/predict', methods=['POST'])
def infer_image():
    # Catch the image file from a POST request
    def_
    

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'