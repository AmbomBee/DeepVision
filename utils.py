import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

def load_img(filename):
    img = nib.load(filename)
    fdata = img.get_fdata()
    header = img.header
    return fdata, header

def show_image(img, ctype):
    plt.figure(figsize=(10, 10))
    if ctype == 'rgb':
        plt.imshow(img)
    elif ctype == 'gray':
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()
    
def get_dataloader(dataset, batch_size, num_GPU):
    """
    Returns the specific dataloader to the batch size
    """
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                       shuffle=True, num_workers=0*num_GPU)
