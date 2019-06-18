import nibabel as nib
from matplotlib import pyplot as plt
import cv2
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.data import DataLoader
import scipy.misc 
from visdom import Visdom

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
    
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, x_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel=x_name,
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

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
             val_indices, test_indices, net, optimizer, criterion, iter_num=None):
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
                'criterion': criterion,
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
                'criterion': criterion,
                'net' : net
                }, filename)
        print('Network saved to ' + filename)
