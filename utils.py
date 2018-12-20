# Imports
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
from PIL import Image
import json

def process_image(image):

    # Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    img = img.resize((256,256))
    img = img.crop((0,0,224,224))

    means = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # convert color to floats between 0 and 1 
    np_image = np.array(img)
    np_image_float = np.array(img)/255
    np_image_normalized = (np_image_float-means)/std
    np_image_transposed = np_image_normalized.transpose((2,0,1))

    return torch.from_numpy(np_image_transposed)

def imshow(image, ax=None, title=None):

    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

 
    image = np.clip(image, 0, 1)    # Clipping image between 0 and 1 

    ax.imshow(image)

    return ax

def display_image_and_chart(img, probs, labels):

    fig, (ax1, ax2) = plt.subplots(figsize=(4,6), nrows=2)
    y_pos = np.arange(len(probs))

    #print image
    image = process_image(img)
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax1.imshow(image)
    ax1.set_title(labels[0].title())
    ax1.axis('off')

    #draw chart
    ax2.barh(y_pos, probs, align='center',color='blue')
    ax2.set_aspect(0.1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_title('Flower Probability')
    ax2.set_xlim(0, 1.1)
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.set_xlabel('Probability')

    plt.tight_layout()
    plt.show()
    print('Chart printed')    

def load_cat_names(cat_to_name):
    with open(cat_to_name, 'r') as f:
        categories = json.load(f)
        return categories


