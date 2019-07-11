from test import test
import dataset
import utils
import network

import torch
import numpy as np  
import time
from tqdm.auto import tqdm

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.manual_seed(1)
np.random.seed(1)

if __name__ == "__main__":
    start = time.time()
    
    plotter = utils.VisdomLinePlotter(env_name='Plots')
    path_to_net = '/home/master/DeepVision/02Evaluation/Metrics/20190711/Cycle_num1/cv_1_99net.pt'
    dirs = '/home/master/DeepVision/01Data/MICCAI_BraTS_2019_Data_Training/HGG/'
    
    checkpoint = torch.load(path_to_net)
    net = checkpoint['net']
    test_indices = checkpoint['test_indices']
    batch_size = 6#checkpoint['batch_size']
    random_state=666
    
    device = torch.device('cuda:0' 
                          if torch.cuda.is_available() else 'cpu')
    num_GPU = torch.cuda.device_count()
    MRI_dataset_test = dataset.MRIDataset(dirs, test_indices)
    test_length = len(MRI_dataset_test)
    test_loader = utils.get_dataloader(MRI_dataset_test, 
                                         batch_size, num_GPU)
    metrics = test(net, test_loader, test_length, device, plotter, batch_size)
    print('Final Metrics: /n', flush=True)
    score, class_iou = metrics.get_scores()
    for k, v in score.items():
        print(k, v, flush=True)
    for k, v in class_iou.items():
        print('Class {} IoU: {}'.format(k, v), flush=True)

    print('Whole run took ', time.time()-start, flush=True)
    print('Done!', flush=True)
