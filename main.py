import glob
import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch
from tqdm.auto import tqdm
import time

import utils
from train import train

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(1)
np.random.seed(1)

if __name__ == "__main__":
    start = time.time()
    
    # cycle_num for different jobs 
    # for different jobs use different optimizer
    # cycle_num % 2 == 0 --> SGD
    # cycle_num % 2 != 0 --> Adam
    cycle_num = 0

    dirs = '/export/home/dv/dv002/DL/project/DeepVision/dataset_pattern/'
    path_to_net = '/export/home/dv/dv002/DL/project/DeepVision/Network/' 

    train(cycle_num, dirs, path_to_net)
    
    print('Whole run took ', time.time()-start, flush=True)
    print('Done!', flush=True)
