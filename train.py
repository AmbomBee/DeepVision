from network import U_Net
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


def train(cycle_num, dirs, path_to_net, plotter, batch_size=2, test_split=0.3, random_state=666, epochs=5, 
          learning_rate=0.001, momentum=0.9, num_folds=5):
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
       
        sigmoid = nn.Sigmoid()
        criterion = nn.BCELoss()
    
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
                   
                running_acc = 0.0
                running_loss = 0.0
                ra_mini = 0.0
                rl_mini = 0.0
                # Iterate over data.
                for i, data in tqdm(enumerate(dataloaders[phase]), desc='Dataiteration'):
                    if i % 500 == 0:
                        print('Number of Iteration [{}/{}]'.format(i+1, int(datalengths[phase]/batch_size)))
                    
                    # get the inputs
                    inputs = data['mri_data'].type(torch.FloatTensor).to(device)
                    segmentations = data['seg'].type(torch.FloatTensor).to(device)
                    
                    # Clear all accumulated gradients
                    optimizer.zero_grad()

                    # Predict classes using inputs and coords from the train set
                    outputs = net(inputs)
                    print('outputs: ', outputs.data)
                    outputs = outputs.view(batch_size, -1)
                    #print('outputs after flatten: ', outputs.size())
                    segmentations = segmentations.view(batch_size, -1)
                    #print('segmentations after flatten: ', segmentations.size())
                    # Compute the loss based on the predictions and 
                    # actual segmentation
                    
                    #print('outputs shape: ', outputs.size())
                    #print('segmentation type: ', type(segmentations))
                    #print('segmentation shape: ', segmentations.size())
                    #print('segmentation dtype: ', segmentations.type())
                    #print('outputs dtype: ', outputs.type())
                    print('min outputs: ', outputs.min())
                    print('max outputs: ', outputs.max())
                    prediction = sigmoid(outputs)
                    print('min pred: ', prediction.min())
                    print('max pred: ', prediction.max())
                    loss = criterion(prediction, segmentations)

                    # Compute the loss based on the predictions and 
                    # actual segmentation
                    loss = criterion(outputs, segmentations)

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
                    running_loss += loss.item()
                    rl_mini += loss.item()

<<<<<<< HEAD
                    running_acc += torch.sum(prediction == segmentations.data).item() #IS THAT CORRECT???????!!!!!!!!!!!!!!!!!!!!!!!!
                    ra_mini += torch.sum(prediction == segmentations.data).item()
                    # print loss and acc and save net every 100 mini-batches
                    if i % 100 == 0:
                        loss_av = rl_mini / (batch_size*100)
                        acc_av = ra_mini / (batch_size*100)
                        print(phase, ' [{}, {}] loss: {}'.format(epoch + 1, i + 1, 
                                                         loss_av))
                        
                        print(phase, ' [{}, {}] acc: {}'.format(epoch + 1, i + 1, 
                                                         acc_av))
                        rl_mini = 0.0
                        ra_mini = 0.0

                        utils.save_net(path_to_net, batch_size, epoch, cycle_num, 
                               train_indices, val_indices, test_indices, net, 
                               optimizer, criterion, iter_num=i)
                        if phase == 'train':
                            plotter.plot('loss', 'itr', 'train', 'Class Loss', (epoch+1)*i,loss_av)
                            plotter.plot('acc', 'train', 'itr', 'Class Acc', (epoch+1)*i,acc_av)

                        if phase == 'val':
                            plotter.plot('loss', 'val', 'itr', 'Class Loss', (epoch+1)*i,loss_av)
                            plotter.plot('acc', 'val', 'itr', 'Class Acc', (epoch+1)*i,acc_av)                         
                    
                    
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
        
        # Save net after every cross validation cycle
        utils.save_net(path_to_net, batch_size, epochs, cycle_num, 
                       train_indices, val_indices, test_indices, net, 
                       optimizer, criterion)
