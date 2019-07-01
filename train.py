from utils import *
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

def train(cycle_num, dirs, path_to_net, batch_size=16, test_split=0.3, 
          random_state=666, epochs=5, learning_rate=0.0001, momentum=0.9, 
          num_folds=5, num_slices=155):
    """
    Applies training on the network
        Args: 
            cycle_num (int): number of cycle in n-fold (num_folds) cross validation
            dirs (string): path to dataset subject directories 
            path_to_net (string): path to directory where to save network
            batch_size - default (int): batch size
            test_split - default (float): percentage of test split 
            random_state - default (int): seed for k-fold cross validation
            epochs - default (int): number of epochs
            learning_rate - default (float): learning rate 
            momentum - default (float): momentum
            num_folds - default (int): number of folds in cross validation
            num_slices - default (int): number of slices per volume
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
        loss_history_train = []
        IoU_history_train = []
        loss_history_val = []
        IoU_history_val = []
        for epoch in range(epochs):
            print('Epoch: ', epoch+1, flush=True)
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                print('Phase: ', phase, flush=True)
                start = time.time()
                if phase == 'train':
                    # Set model to training mode
                    net.train(True)
                elif phase == 'val':
                    # Set model to evaluate mode
                    net.train(False)
                
                running_loss = 0.0
                rl_mini = 0.0
                running_IoU = 0.0
                r_IoU_mini = 0.0
                # Iterate over data.
                for i, data in enumerate(dataloaders[phase]):
                    if i % 10 == 0:
                        print('Number of Iteration [{}/{}]'.format(i+1, 
                        int(datalengths[phase]/batch_size)), flush=True)
                    
                    # get the inputs
                    inputs = data['mri_data'].type(torch.FloatTensor).to(device)
                    segmentations = data['seg'].type(torch.FloatTensor)
                    segmentations = one_hot(segmentations).to(device)
                    subject_slice_path = data['subject_slice_path']
                    # Clear all accumulated gradients
                    optimizer.zero_grad()
                    # Predict classes using inputs from the train set
                    outputs = net(inputs)
                    # Compute the loss based on the predictions and 
                    # actual segmentation
                    softmax = nn.Softmax2d()
                    predictions = softmax(outputs)
                    loss = dice_loss(predictions, segmentations)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # Backpropagate the loss
                        loss.backward(retain_graph=True)
                        # Adjust parameters according to the computed 
                        # gradients 
                        # -- weight update
                        optimizer.step()
                    # track loss and IoU over epochs and all 500 iterations
                    iou = IoU(predictions, segmentations).item()
                    running_IoU += iou
                    r_IoU_mini += iou
                    running_loss += loss.item()
                    rl_mini += loss.item()
                    if (i + 1) % 10 == 0:
                        loss_av = rl_mini / (batch_size*10)
                        IoU_av = r_IoU_mini / (batch_size*10)
                        print(phase, ' [{}, {}] loss: {}'.format(epoch + 1, i + 1, 
                                                         loss_av), flush=True)
                        print(phase, ' [{}, {}] IoU: {}'.format(epoch + 1, i + 1, 
                                                         IoU_av), flush=True)
                        if phase == 'train':
                            loss_history_train.append(loss_av)
                            IoU_history_train.append(IoU_av)
                            plot_history(phase, epoch, i, path_to_net, 
                                         loss_history_train, 
                                         IoU_history_train)
                        if phase == 'val':
                            loss_history_val.append(loss_av)
                            IoU_history_val.append(IoU_av)
                            plot_history(phase, epoch, i, path_to_net, 
                                         loss_history_val, 
                                         IoU_history_val)
                        rl_mini = 0.0
                        r_IoU_mini = 0.0
                        save_net(path_to_net, batch_size, epoch, cycle_num, train_indices, 
                                 val_indices, test_indices, net, optimizer, iter_num=i)
                        save_output(i, path_to_net, subject_slice_path, 
                                    outputs.detach().cpu().numpy(), 
                                    segmentations.detach().cpu().numpy())
                    del loss
                print('Phase {} took {} s for whole {}set!'.format(phase, 
                      time.time()-start, phase), flush=True)
                # Compute the average IoU and loss over all inputs
                running_loss = running_loss / datalengths[phase]
                running_IoU = running_IoU / datalengths[phase]
                if phase == 'train':
                    print ('Epoch [{}/{}], Train_loss: {:.4f}, Train_IoU: {:.2f}' 
                           .format(epoch+1, epochs, running_loss, 
                                   running_IoU), flush=True)
                elif phase == 'val':
                    print ('Epoch [{}/{}], Val_loss: {:.4f}, Val_IoU: {:.2f}' 
                           .format(epoch+1, epochs, running_loss, 
                                   running_IoU), flush=True)
            # Call the learning rate adjustment function after every epoch
            scheduler.step(running_loss)
    # save network after training
    save_net(path, batch_size, epochs, cycle_num, train_indices, 
             val_indices, test_indices, net, optimizer, iter_num=None)
