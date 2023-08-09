import plotly.express as px
from preprocessing import decompose_emodb
import numpy as np
from collections import defaultdict

import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from PIL import Image

import pandas as pd
import os
import sklearn
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
import pytorchtools
from PIL import Image
import preprocessing
import torch as torch
from torchvision import transforms
import torch
from tqdm import tqdm
import scoring

# %% [markdown]
# ### Exploratory Data Analysis

# %%
audio_dataset_path = "../data/raw/wav/"
image_dataset_path = "../data/preprocessed/images/jpeg/"
dataset_exists = os.path.exists(image_dataset_path)
if not dataset_exists:
    # Create a new directory because it does not exist
    os.makedirs(image_dataset_path)
    print("directory is created!")

dataset_summary = preprocessing.decompose_emodb(audio_dataset_path)
preprocessing.create_audio_spectrogram(
    image_dataset_path, audio_dataset_path, dataset_summary
)
dataset_summary = preprocessing.decompose_emodb(image_dataset_path)


# %%
labels = dataset_summary.labels.unique()
le = sklearn.preprocessing.LabelEncoder()
targets = le.fit_transform(labels)
le.inverse_transform(targets)

# %%
preprocessing.label_encoder_to_json(le, "labels.json")
# %%
dataset_summary.replace(labels, le.fit_transform(labels), inplace=True)
from torch.utils.data import Dataset, DataLoader

# 0.8 * 0.125 gives 0.1 for validation
train_df, test_df = train_test_split(
    dataset_summary, test_size=0.2, stratify=dataset_summary.labels
)
train_df, valid_df = train_test_split(
    train_df, test_size=0.125, stratify=train_df.labels
)

# %%
train_df.reset_index(inplace=True, drop=True)
test_df.reset_index(inplace=True, drop=True)
valid_df.reset_index(inplace=True, drop=True)

# %%
resize = T.Compose([T.Resize((256, 256)), T.ToTensor()])

train_dataset = preprocessing.EmoDataset(train_df, transform=resize)
test_dataset = preprocessing.EmoDataset(test_df, transform=resize)
valid_dataset = preprocessing.EmoDataset(valid_df, transform=resize)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=32, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=32, shuffle=True
)
valid_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset, batch_size=32, shuffle=True
)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## USE APPLE M1 GPU
try:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
except AttributeError:
    print(device)

# model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
model = torch.hub.load("pytorch/vision:v0.10.0", "alexnet", pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)


##REPLACE LAST LAYER
num_labels = len(dataset_summary["labels"].unique())
model.classifier[6] = torch.nn.Linear(4096, num_labels)
# Freeze the gradients of all of the layers in the features (convolutional) layers
# for param in model.features.parameters():
#     param.requires_grad = False


params = {"learning_rate": 1e-5, "epochs": 40, "patience": 2}
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
running_loss = 0
losses = []

# %%
model.to(device)

# %%
# to track the training loss as the model trains
train_losses = []
# to track the validation loss as the model trains
valid_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
avg_valid_losses = []

# %%
size = len(train_loader.dataset)
print(size)
if not os.path.exists("../models/"):
    os.mkdir("../models/")
# initialize the early_stopping object
early_stopping = pytorchtools.EarlyStopping(
    patience=params["patience"], verbose=True, model_path="../models/alexnet_emodb.pt"
)

for epoch in range(params["epochs"]):
    #######TRAIN MODEL########
    epochs_loss = 0

    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images).to(device)
        loss = criterion(outputs, labels)

        # Backprpagation and optimization
        optimizer.zero_grad()
        loss.backward()
        # calculate train_loss
        train_losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
        optimizer.step()
    ##########################
    #####TEST MODEL#######
    ##########################
    accuracy = scoring.testAccuracy(model, test_loader, device)
    ##########################
    #####VALIDATE MODEL#######
    ##########################
    model.eval()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images).to(device)
        loss = criterion(outputs, labels)
        valid_losses.append(loss.item())

    # print training/validation statistics
    # calculate average loss over an epoch
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    # print(train_loss)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)

    print_msg = (
        f"accuracy: {accuracy:.3f} "
        + f"train_loss: {train_loss:.3f} "
        + f"valid_loss: {valid_loss:.3f}"
    )

    print(print_msg)

    # clear lists to track next epoch
    train_losses = []
    valid_losses = []

    early_stopping(valid_loss, model)
    print(epoch)

    if early_stopping.early_stop:
        print("Early stopping")
        break


#


errs = scoring.error_analysis(test_loader, model, le, device)
halt = avg_valid_losses.index(min(avg_valid_losses))
