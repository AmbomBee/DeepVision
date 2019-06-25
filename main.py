import glob
import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch
from tqdm.auto import tqdm
import time

import utils
#from dataset import MRIDataset, ToTensor
from train import train

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.manual_seed(1)
np.random.seed(1)

if __name__ == "__main__":
    start = time.time()
    
    cycle_num = 0

    dirs = '/export/home/dv/dv002/DL/project/DeepVision/dataset_pattern/'
    path_to_net = '/export/home/dv/dv002/DL/project/DeepVision/Network/'
    path_for_saving_SR = '/export/home/dv/dv002/DL/project/DeepVision/output_images/' 

    '''FOR ANN-SOPHIE'''
    '''
    dirs = '/media/hilkert/Hilkert/DeepVision/05Data/BraTS/MICCAI_BraTS_2018_Data_Training/HGG/'
    path_to_net = '/media/hilkert/Hilkert/DeepVision/06Network/'
    '''
    train(cycle_num, dirs, path_to_net, path_for_saving_SR)
    
    print('Whole run took ', time.time()-start)
    print('Done!')
