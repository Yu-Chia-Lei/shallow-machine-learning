"""
Image Deblurring using Convolutional Neural Networks and Deep Learning

Part 2: Model definitions and utility functions

This file contains 
1. Three CNN models: DeblurCNN model, DeblurCNN_RES model 
    and the DeblurSuperResCNN model.

2. A DeblurDataset class: a custom dataset class for loading images. 
3. The psnr function for computing Peak Signal to Noise Ratio (PSNR).
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torch
import cv2
import math
import numpy as np
from torch.utils.data import Dataset


class DeblurCNN(nn.Module):
    '''
    DeblurCNN model: original SRCNN model from paper
    '''
    def __init__(self):
        super(DeblurCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=9, stride=(1, 1), padding=(2, 2) 
        ) # 3 input image channel, 64 output channels, 9x9 square convolution
        self.conv2 = nn.Conv2d(
            64, 32, kernel_size=1, stride=(1, 1), padding=(2, 2)
        ) # 64 input image channel, 32 output channels, 1x1 square convolution
        self.conv3 = nn.Conv2d(
            32, 3, kernel_size=5, stride=(1, 1), padding=(2, 2)
        ) # 32 input image channel, 3 output channels, 5x5 square convolution

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x
    
# SRCNN model: modified model from paper
class DeblurCNN_RES(nn.Module):
    '''
    DeblurCNN_RES model: originate from the Residual SRCNN model from paper
    '''
    def __init__(self):
        super(DeblurCNN_RES, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=9, stride=(1, 1), padding=(2, 2) 
        ) # 3 input image channel, 64 output channels, 9x9 square convolution
        self.conv2 = nn.Conv2d(
            64, 32, kernel_size=1, stride=(1, 1), padding=(2, 2)
        ) # 64 input image channel, 32 output channels, 1x1 square convolution
        self.conv3 = nn.Conv2d(
            32, 3, kernel_size=5, stride=(1, 1), padding=(2, 2)
        ) # 32 input image channel, 3 output channels, 5x5 square convolution

    def forward(self, x):
        identity = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        out = torch.add(x, identity) # residual connection
        return out


class DeblurSuperResCNN(nn.Module):
    '''
    DeblurSuperResCNN model: originate from the Deblurring + SRCNN model from paper
    '''
    def __init__(self):
        super(DeblurSuperResCNN, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        # Third Convolutional Layer (Feature Concatenation)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        # Output Layer (Reconstruction)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # First Layer
        out1 = self.relu1(self.conv1(x))

        # Second Layer
        out2 = self.relu2(self.conv2(out1))

        # Feature Concatenation
        concat = torch.cat((out1, out2), dim=1)  # Concatenate along channel dimension

        # Third Layer
        out3 = self.relu3(self.conv3(concat))

        # Output Layer
        out_final = self.sigmoid(self.conv4(out3))

        return out_final



class DeblurDataset(Dataset):
    """
    Custom Dataset for loading images for deblurring.
    Args:
        blur_path: directory of blurred images.
        blur_names: List of file names to blurred images.
        sharp_path: directory of sharp images.
        sharp_names: List of file names to sharp images.
        transforms (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, blur_path, blur_names, sharp_path, sharp_names=None, transforms=None):
        self.blur_path = blur_path
        self.sharp_path = sharp_path
        self.X = blur_names
        self.y = sharp_names
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        blur_image = cv2.imread(f"{self.blur_path}/{self.X[i]}")
        blur_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB)  # <-- Add this line
        blur_image = np.array(blur_image, dtype=np.float32)
        blur_image /= 255.
        if self.transforms:
            blur_image = self.transforms(blur_image)
            
        if self.y is not None:
            sharp_image = cv2.imread(f"{self.sharp_path}/{self.y[i]}")
            sharp_image = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2RGB) 
            sharp_image = np.array(sharp_image, dtype=np.float32) 
            sharp_image /= 255.
            sharp_image = self.transforms(sharp_image)
            return (blur_image, sharp_image)
        else:
            return blur_image

def psnr(label, outputs, max_val=1.):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition

    Note that the output and label pixels (when dealing with images) should
    be normalized as the `max_val` here is 1 and not 255.
    """
    # PSNR = 10 * torch.log10(1.0 / torch.mean((outputs - label) ** 2)) 
    # return PSNR.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    diff = outputs - label
    rmse = math.sqrt(np.mean((diff) ** 2)) 
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(max_val / rmse)
        return PSNR

