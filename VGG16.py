# Load the pretrained model from pytorch
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

plt.ion()  

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")


        
# vgg16 = models.vgg16_bn(models.VGG16_BN_Weights)
vgg16 = models.vgg16_bn()
vgg16.load_state_dict(torch.load("vgg16_bn.pth"))
print(vgg16.classifier[6].out_features) # 1000 


# Freeze training for all layers
for param in vgg16.features.parameters():
    param.require_grad = False

# # Newly created modules have require_grad=True by default
# num_features = vgg16.classifier[6].in_features
# features = list(vgg16.classifier.children())[:-1] # Remove last layer
# features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with 4 outputs
# vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
print(vgg16)