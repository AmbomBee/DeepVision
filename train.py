from network import U_Net
from utils import dice_loss
from utils import one_hot
import utils
import dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import numpy as np
import glob
import time
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

def train(cycle_num, dirs, path_to_net, path_for_saving_SR, batch_size=20, test_split=0.3, random_state=666, epochs=5, learning_rate=0.001, momentum=0.9, num_folds=5):
    """
    Applies training on the network 
    """
    start = time.time()
    print('Setting started')
    
    # Creating data indices
    # number of dirs (one dir per subject) ####times number of slices
    indices = np.arange(len(glob.glob(dirs + '*')))
    
    test_indices, trainset_indices = utils.get_test_indices(indices, 
                                                     test_split)                                                    
    # kfold index generator
    for cv_num, (train_indices, val_indices) in enumerate(utils.get_train_cv_indices(trainset_indices, 
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
            print('Let us use', num_GPU, 
                  'GPUs!')
            net = nn.DataParallel(net)
        net.to(device)
        
        if cycle_num % 2 == 0:
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, 
                                      momentum=momentum)
        else: 
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        
        scheduler = ReduceLROnPlateau(optimizer)#, threshold=1e-12)
        
        print('Setting took: ', time.time()-start)
        print('cv cycle number: ', cycle_num)
        print('\n')
        
        start = time.time()
        print('Start Train and Val loading')
        
        MRIDataset_train = dataset.MRIDataset(dirs, train_indices)
        
        MRIDataset_val = dataset.MRIDataset(dirs, val_indices)
        
        datalengths = {'train': len(MRIDataset_train), 
                       'val': len(MRIDataset_val)}
        dataloaders = {'train': utils.get_dataloader(MRIDataset_train, 
                                               batch_size, num_GPU),
                      'val': utils.get_dataloader(MRIDataset_val, 
                                            batch_size, num_GPU)}
        print('Train and Val loading took: ', time.time()-start)
        
        for epoch in tqdm(range(epochs), desc='Epochs'):
            print('Epoch: ', epoch+1)
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                print('Phase: ', phase)
                
                start = time.time()
                   
                if phase == 'train':
                    # Set model to training mode
                    net.train(True)
                    
                elif phase == 'val':
                    # Set model to evaluate mode
                    net.train(False)  
                   
                #running_acc = 0.0
                running_loss = 0.0
                #ra_mini = 0.0
                rl_mini = 0.0
                r_IoU = 0.0
                r_IoU_mini = 0.0
                # Iterate over data.
                for i, data in tqdm(enumerate(dataloaders[phase]), desc='Dataiteration'):
                    if i % 500 == 0:
                        print('Number of Iteration [{}/{}]'.format(i+1, int(datalengths[phase]/batch_size)))
                    
                    # get the inputs
                    inputs = data['mri_data'].type(torch.FloatTensor).to(device)
                    segmentations = data['seg'].type(torch.FloatTensor).to(device)
                    segmentations = one_hot(segmentations)
                    
                    # Clear all accumulated gradients
                    optimizer.zero_grad()

                    # Predict classes using inputs and coords from the train set
                    outputs = net(inputs)
                    # Compute the loss based on the predictions and 
                    # actual segmentation
                    
                    softmax = nn.Softmax2d()
                    prediction = softmax(outputs) 
                    loss = dice_loss(prediction, segmentations)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # Backpropagate the loss
                        loss.backward()
                        # Adjust parameters according to the computed 
                        # gradients 
                        # -- weight update
                        optimizer.step()
                    # track loss over epochs, mini-batches (500) and 
                    # per iteration
                    iou = utils.IoU(predictions, segmentations)
                    r_IoU += iou
                    r_IoU_mini += iou
                    running_loss += loss.item()
                    rl_mini += loss.item()

                    
                    running_acc += torch.sum(prediction == segmentations.data).item()
                    ra_mini += torch.sum(prediction == segmentations.data).item()
                    # print loss and acc and save net every 100 mini-batches
                    if i % 100 == 0:
                        loss_av = rl_mini / (batch_size*100)
                        #acc_av = ra_mini / (batch_size*100)
                        print(phase, ' [{}, {}] loss: {}'.format(epoch + 1, i + 1, 
                                                         loss_av))
                        
                        rl_mini = 0.0
                        ra_mini = 0.0

                        name_of_file = "Epoch{}_iter{}.png".format(epoch, i)
                        
                        fig, axs = plt.subplots(1, 2, constrained_layout=True)
                        fig.suptitle("SR and GT for {} iteration".format(i+1))
                        axs[0].set_title("segmentation result")
                        axs[1].set_title("ground truth")
                        plt.savefig(path_for_saving_SR + name_of_file)
                        
                        fig = plt.figure()
                        a = fig.add_subplot(1,2,1)
                        a.set_title("segmentation result")
                    
                end = time.time()
                print('Phase {} took {} s for whole {}set!'.format(phase, end-start, 
                                                                   phase))
                
                # Compute the average acc and loss over all inputs
                running_loss = running_loss #/ datalengths[phase]
                running_acc = running_acc / datalengths[phase]
                
                if phase == 'train':
                    print ('Epoch [{}/{}], Train_loss: {:.4f}, Train_acc: {:.2f}' 
                           .format(epoch+1, epochs, running_loss, 
                                   running_acc))
                    
                        
                elif phase == 'val':
                    print ('Epoch [{}/{}], Val_loss: {:.4f}, Val_acc: {:.2f}' 
                           .format(epoch+1, epochs, running_loss, 
                                   running_acc))
                
            # Call the learning rate adjustment function after every epoch
            scheduler.step(running_loss)
