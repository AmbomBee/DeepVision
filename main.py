import glob
import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch
from tqdm.auto import tqdm

import utils
from dataset import MRIDataset, ToTensor



if __name__ == "__main__":
    
    dirs = '/home/hilkert/Dokumente/Uni/DeepVision/Project/05Data/BraTS/MICCAI_BraTS_2018_Data_Training/HGG/'
    idc = np.arange(len(glob.glob(dirs + '*')))
    mri_dataset = MRIDataset(dirs, idc, transform=ToTensor())
    datalength = len(mri_dataset)
    batch_size = 1
    device = torch.device('cuda:0' 
                        if torch.cuda.is_available() else 'cpu')
    num_GPU = torch.cuda.device_count()
    if  num_GPU > 1:
        print('Let us use ', num_GPU, ' GPUs!')
    dataloader = utils.get_dataloader(mri_dataset, batch_size, num_GPU)
    
    for i, data in tqdm(enumerate(dataloader), desc='Dataiteration'):
        if i % 500 == 0:
            print('Number of Iteration [{}/{}]'.format(i+1, datalength // batch_size))
        
        # get the inputs
        inputs = data['mri_data'].to(device)
        segs = data['seg'].type(torch.LongTensor).to(device)
        print(segs.size())
        plt.imshow(segs)
        for seg in segs:
            plt.imshow(seg)
