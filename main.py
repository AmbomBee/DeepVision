import glob
import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np
import torch
import time
import sys

import utils
#from retrain import retrain
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
    cycle_num = int(sys.argv[1])
    plotter = utils.VisdomLinePlotter(env_name='Plots')
    dirs = '/home/master/DeepVision/01Data/MICCAI_BraTS_2019_Data_Training/HGG/'
    path_to_net = '/home/master/DeepVision/02Evaluation/Cycle_num' + str(cycle_num) + '/' 
    train(cycle_num, dirs, path_to_net, plotter)
    #retrain(cycle_num, dirs, path_to_net, plotter)
    print('Whole run took ', time.time()-start, flush=True)
    print('Done!', flush=True)
