import nibabel as nib
from matplotlib import pyplot as plt
import cv2
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.data import DataLoader
import scipy.misc

def save_output(i, path_to_net, path, SR, GT):
    torch.save({'SR': SR, 'GT': GT, 'path': path}, path_to_net + 'output' + str(i) + '.pt')
    
def plot_history(path_to_net, loss_history, IoU_history):
    fig, ax = plt.subplots(1,2)
    ax[0].plot(loss_history)
    ax[0].set_title('Loss')
    ax[1].plot(IoU_history)
    ax[1].set_title('IoU')
    plt.savefig(path_to_net + '.svg')
    plt.close()
    
def load_img(filename):
    img = nib.load(filename)
    fdata = img.get_fdata()
    header = img.header
    return fdata, header

def show_image(img, ctype):
    plt.figure(figsize=(10, 10))
    if ctype == 'rgb':
        plt.imshow(img)
    elif ctype == 'gray':
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()
    
def get_train_cv_indices(indices, num_folds, random_state):
    """
    Creates a generator for trainset_indices and test_indices
    Trainset includes the indices for validation
    """
    kf = KFold(n_splits=num_folds,random_state=random_state)
    return kf.split(np.zeros(len(indices)), np.zeros(len(indices)))

def get_test_indices(indices, test_split,):
    """
    Returns test indices and training indices
    """
    split = int(np.floor(test_split * len(indices)))
    test_indices, train_indices = indices[:split], indices[split:]
    return test_indices, train_indices

def get_dataloader(dataset, batch_size, num_GPU):
    """
    Returns the specific dataloader to the batch size
    """
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                       shuffle=True, num_workers=0*num_GPU)

def save_net(path, batch_size, epoch, cycle_num, train_indices, 
             val_indices, test_indices, net, optimizer, iter_num=None):
    """
    Saves the networks specific components and the network itselfs 
    to a given path
    """
    if iter_num is not None:
        filename = path + 'cv_' + str(cycle_num) + '_iterin_' + str(epoch) + 'net.pt'
        torch.save({
                'iter_num': iter_num,
                'batch_size': batch_size,
                'epoch': epoch,
                'cycle_num': cycle_num,
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'net' : net
                }, filename)
        print('Network saved to ' + filename)
    else:
        filename = path + 'cv_' + str(cycle_num) + '_' + str(epoch) + 'net.pt'
        torch.save({
                'batch_size': batch_size,
                'epoch': epoch,
                'cycle_num': cycle_num,
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'net' : net
                }, filename)
        print('Network saved to ' + filename)

def one_hot(GT):
    '''
        Returns target by one-hot encoding
        GT: b x W x H
        output: b x c x W x H one hot representation
    '''
    batch_size = GT.shape[0]
    W = GT.shape[1]
    H = GT.shape[2]
    one_hot = torch.empty([batch_size,4,W,H])
    # iterate over batches
    for i in range(batch_size):
        for c in range(4):
            one_hot[i][c]=(GT[i,:,:]==c)
    return one_hot
    
def dice_loss(SR, GT, epsilon=1e-9):
    '''
        Return dice loss for single class lable:
        SR: segmentation result
            batch_size x c x W x H
        GT: ground truth
            batch_size x c x W x H
        epsilon: used for numerical stability to avoid devide by zero errors
        Dice = 2*|Intersection(A,B)| / (|A| + |B|)
    '''
    # count memberwise product of SR and GT, then sum by axis (2,2)
    numerator = 2*torch.sum(torch.mul(SR,GT), (2,3)) 
    SR_n = torch.mul(SR,SR)
    GT_n = torch.mul(GT,GT) 
    denominator = torch.sum(SR_n + GT_n, (2,3))
    # average dice over classes and batches
    ret = 1 - torch.mean(numerator/(denominator+epsilon))
    return torch.tensor(ret, requires_grad=True, dtype = torch.float)

def IoU(SR, GT, epsilon = 1e-9):
    '''
        IoU = |Intersection(SR,GT)| / (|SR| + |GT| - |Intersection(SR,GT)|)
    '''
    numerator = torch.sum(torch.mul(SR,GT), (2,3)) 
    SR_n = torch.mul(SR,SR)
    GT_n = torch.mul(GT,GT) 
    denominator = torch.sum(torch.add(SR_n, GT_n), (2,3)) - numerator
    return torch.mean(numerator / (denominator + epsilon))
