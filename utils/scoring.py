import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from sklearn.metrics import confusion_matrix


def testAccuracy(model, test_loader, device):
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            # run the model on the test set to predict labels
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).to(device)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    # compute the accuracy over all test images
    accuracy = accuracy / total
    return accuracy


def generate_confusion_matrix(y_true, y, classes):
    cf_matrix = confusion_matrix(y_true, y)

    ax = plt.subplot()
    sns.heatmap(cf_matrix, annot=True)

    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)


def error_analysis(test_loader, model, label_encoder, device):
    model.eval()
    preds = []
    true_labels = []
    for images, labels in test_loader:

        data, target = images.to(device), labels.to(device)
        output = model(data)  # shape = torch.Size([batch_size, 10])
        pred = output.argmax(
            dim=1, keepdim=True
        )  # pred will be a 2d tensor of shape [batch_size,1]

        preds.append(pred.flatten().to("cpu").numpy())
        true_labels.append(labels.flatten().numpy())

    #### GET MISIDENTIFIED EXAMPLE
    all_preds = np.concatenate(preds, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    df = pd.DataFrame({"expected": true_labels, "predicted": all_preds})

    ### DECODE LABELS
    for col in df:
        df[col] = label_encoder.inverse_transform(df[col])

    return df
