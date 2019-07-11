import utils
from metrics import *
from utils import *

import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
import csv

def test(net, test_loader, datalength, device, plotter, batch_size, n_classes=4):
    """
    Applies testing on the network
    """
    phase = 'test'
    net.to(device)
    net.eval()
    with torch.no_grad():
        # Setup Metrics
        running_metrics_test = runningScore(n_classes)
        for i, data in tqdm(enumerate(test_loader), desc='Testdata'):
            if (i + 1) % 100 == 0:
                print('Number of Iteration [{}/{}]'.format(i+1, 
                int(datalength/batch_size)), flush=True)
            # get the inputs
            inputs = data['mri_data'].to(device)
            GT = data['seg'].to(device)
            subject_slice_path = data['subject_slice_path']
            # Predict classes using inputs from the train set
            SR = net(inputs)
            predictions = SR.data.max(1)[1].cpu().numpy()
            GT_cpu = GT.data.cpu().numpy()
            running_metrics_test.update(GT_cpu, predictions)
            if (i + 1) % 100 == 0:
                score, class_iou = running_metrics_test.get_scores()
                for k, v in score.items():
                    plotter.plot(k, 'itr', phase, k, i, v)
                for k, v in class_iou.items():
                    print('Class {} IoU: {}'.format(k, v), flush=True)
                    plotter.plot(str(k) + ' Class IoU', 'itr', phase, 
                                 str(k) + ' Class IoU',  i, v)
                save_output(i, './', subject_slice_path, 
                                SR.data.cpu().numpy(), GT_cpu)
    return running_metrics_test
