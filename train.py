from utils import *
from metrics import runningScore, averageMeter
import dataset
from network import U_Net

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import numpy as np
import glob
import time
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def train(cycle_num, dirs, path_to_net, plotter, batch_size=12, test_split=0.3, 
          random_state=666, epochs=100, learning_rate=0.0001, momentum=0.9, 
          num_folds=5, num_slices=155, n_classes=4):
    """
    Applies training on the network
        Args: 
            cycle_num (int): number of cycle in n-fold (num_folds) cross validation
            dirs (string): path to dataset subject directories 
            path_to_net (string): path to directory where to save network
            plotter (callable): visdom plotter
            batch_size - default (int): batch size
            test_split - default (float): percentage of test split 
            random_state - default (int): seed for k-fold cross validation
            epochs - default (int): number of epochs
            learning_rate - default (float): learning rate 
            momentum - default (float): momentum
            num_folds - default (int): number of folds in cross validation
            num_slices - default (int): number of slices per volume
            n_classes - default (int): number of classes (regions)
    """
    print('Setting started', flush=True)
    
    # Creating data indices
    # arange len of list of subject dirs 
    indices = np.arange(len(glob.glob(dirs + '*')))
    test_indices, trainset_indices = get_test_indices(indices, 
                                                     test_split)                                                    
    # kfold index generator
    for cv_num, (train_indices, val_indices) in enumerate(get_train_cv_indices(trainset_indices, 
                                                         num_folds, 
                                                         random_state)):
        # splitted the 5-fold CV in 5 jobs
        if cv_num != int(cycle_num):
            continue
            
        net = U_Net()
        device = torch.device('cuda:0' 
                        if torch.cuda.is_available() else 'cpu')
        num_GPU = torch.cuda.device_count()
        if  num_GPU > 1:
            print('Let us use {} GPUs!'.format(num_GPU), flush=True)
            net = nn.DataParallel(net)
        net.to(device)
        criterion = nn.CrossEntropyLoss()
        if cycle_num % 2 == 0:
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, 
                                      momentum=momentum)
        else: 
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        
        scheduler = ReduceLROnPlateau(optimizer, threshold=1e-6, patience=0)

        print('cv cycle number: ', cycle_num, flush=True)
        start = time.time()
        print('Start Train and Val loading', flush=True)
        
        MRIDataset_train = dataset.MRIDataset(dirs, train_indices)
        
        MRIDataset_val = dataset.MRIDataset(dirs, val_indices)
        
        datalengths = {'train': len(MRIDataset_train), 
                       'val': len(MRIDataset_val)}
        dataloaders = {'train': get_dataloader(MRIDataset_train, 
                                               batch_size, num_GPU),
                      'val': get_dataloader(MRIDataset_val, 
                                            batch_size, num_GPU)}
        print('Train and Val loading took: ', time.time()-start, flush=True)
        # make loss and acc history for train and val separatly
        # Setup Metrics
        running_metrics_val = runningScore(n_classes)
        running_metrics_train = runningScore(n_classes)
        val_loss_meter = averageMeter()
        train_loss_meter = averageMeter()
        itr = 0
        iou_best = 0.
        for epoch in tqdm(range(epochs), desc='Epochs'):
            print('Epoch: ', epoch+1, flush=True)
            phase = 'train'
            print('Phase: ', phase, flush=True)
            start = time.time()
            # Set model to training mode
            net.train()
            # Iterate over data.
            for i, data in tqdm(enumerate(dataloaders[phase]), desc='Data Iteration ' + phase):
                if (i + 1) % 100 == 0:
                    print('Number of Iteration [{}/{}]'.format(i+1, 
                    int(datalengths[phase]/batch_size)), flush=True)
                # get the inputs
                inputs = data['mri_data'].to(device)
                GT = data['seg'].to(device)
                subject_slice_path = data['subject_slice_path']
                # Clear all accumulated gradients
                optimizer.zero_grad()
                # Predict classes using inputs from the train set
                SR = net(inputs)
                # Compute the loss based on the predictions and 
                # actual segmentation
                loss = criterion(SR, GT)
                # Backpropagate the loss
                loss.backward()
                # Adjust parameters according to the computed 
                # gradients 
                # -- weight update
                optimizer.step()
                # Trake and plot metrics and loss, and save network
                predictions = SR.data.max(1)[1].cpu().numpy()
                GT_cpu = GT.data.cpu().numpy()
                running_metrics_train.update(GT_cpu, predictions)
                train_loss_meter.update(loss.item(), n=1)
                if (i + 1) % 100 == 0:
                    itr += 1
                    score, class_iou = running_metrics_train.get_scores()
                    for k, v in score.items():
                        plotter.plot(k, 'itr', phase, k, itr, v)
                    for k, v in class_iou.items():
                        print('Class {} IoU: {}'.format(k, v), flush=True)
                        plotter.plot(str(k) + ' Class IoU', 'itr', phase, 
                                     str(k) + ' Class IoU',  itr, v)
                    print('Loss Train', train_loss_meter.avg, flush=True)
                    plotter.plot('Loss', 'itr', phase, 'Loss Train', 
                                 itr, train_loss_meter.avg)
            print('Phase {} took {} s for whole {}set!'.format(phase, 
                  time.time()-start, phase), flush=True)
            
            # Validation Phase
            phase = 'val'
            print('Phase: ', phase, flush=True)
            start = time.time()
            # Set model to evaluation mode
            net.eval()
            start = time.time()
            with torch.no_grad():
                # Iterate over data.
                for i, data in tqdm(enumerate(dataloaders[phase]), desc='Data Iteration ' + phase):
                    if (i + 1) % 100 == 0:
                        print('Number of Iteration [{}/{}]'.format(i+1, 
                              int(datalengths[phase]/batch_size)), flush=True)
                    # get the inputs
                    inputs = data['mri_data'].to(device)
                    GT = data['seg'].to(device)
                    subject_slice_path = data['subject_slice_path']
                    # Clear all accumulated gradients
                    optimizer.zero_grad()
                    # Predict classes using inputs from the train set
                    SR = net(inputs)
                    # Compute the loss based on the predictions and 
                    # actual segmentation
                    loss = criterion(SR, GT)
                    # Trake and plot metrics and loss
                    predictions = SR.data.max(1)[1].cpu().numpy()
                    GT_cpu = GT.data.cpu().numpy()
                    running_metrics_val.update(GT_cpu, predictions)
                    val_loss_meter.update(loss.item(), n=1)
                    if (i + 1) % 100 == 0:
                        itr += 1
                        score, class_iou = running_metrics_val.get_scores()
                        for k, v in score.items():
                            plotter.plot(k,
                                         'itr', phase, k,  itr, v)
                        for k, v in class_iou.items():
                            print('Class {} IoU: {}'.format(k, v), flush=True)
                            plotter.plot(str(k) + ' Class IoU', 'itr', 
                                         phase, str(k) + ' Class IoU',  itr, v)
                        print('Loss Val', val_loss_meter.avg, flush=True)
                        plotter.plot('Loss ', 'itr', phase, 'Loss Val', 
                                     itr, val_loss_meter.avg)
                if (epoch + 1) % 10 == 0:
                    if score['Mean IoU'] > iou_best:
                        save_net(path_to_net, batch_size, epoch, cycle_num, train_indices, 
                                 val_indices, test_indices, net, optimizer)
                        iou_best = score['Mean IoU']
                    save_output(epoch, path_to_net, subject_slice_path, 
                                SR.data.cpu().numpy(), GT_cpu)
                print('Phase {} took {} s for whole {}set!'.format(phase, 
                      time.time()-start, phase), flush=True)
            # Call the learning rate adjustment function after every epoch
            scheduler.step(val_loss_meter.avg)
    # save network after training
    save_net(path_to_net, batch_size, epochs, cycle_num, train_indices, 
             val_indices, test_indices, net, optimizer, iter_num=None)
