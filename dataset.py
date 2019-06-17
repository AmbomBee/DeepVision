import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import glob

class MRIDataset(Dataset):
    """
    MRI Data set
    """
    
    def __init__(self, img_dir, idc, num_slices=155, transform=None):
        """
        Args:
            label_file (string): Path to the csv file with labels.
            nii_dir (string): directory all the nii files (fMRI data).
            random_state (object of np.random.RandomState class)
            idc (list of ints): list of indices used for training, 
                validation or testing
        """
        
        self.idc = idc
        self.img_dir = img_dir
        self.nii_dir = np.asarray(glob.glob(self.img_dir + '/*'))[self.idc]
        self.nii_per_dir = len(glob.glob(self.nii_dir[0] + '/*'))
        self.num_slices = num_slices
        self.transform = transform
                    
    def __len__(self):
        return len(self.nii_dir) * self.nii_per_dir * num_slices
    
    def __get_img__(self, img_name):
        """
        Returns the segmentation image
        """
        img = nib.load(img_name)
        img_data = img.get_fdata()
        return img_data
    
    def __getitem__(self, idx):
        """
        supports the indexing of MRIDataset
        """
        # indices to subject idx
        sub_idx = idx // (self.nii_per_dir * self.num_slices)
        slice_idx = (idx // self.nii_per_dir) % self.num_slices
        path_nii = glob.glob(self.nii_dir[sub_idx] + '/*')
        mri_data = []
        for i, path in enumerate(path_nii):
            if i == 1:
                
                #####################################################################
                seg = self.__get_img__(path)[]
            else:
                mri_data.append(self.__get_img__(path))
        sample = {'mri_data': np.asarray(mri_data), 'seg': np.asarray(seg)}

        if self.transform:
            sample = self.transform(sample)
            
        return sample


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors
    """
    def __call__(self, sample):
        mri_data, seg = sample['mri_data'], sample['seg']
        return {'mri_data': torch.from_numpy(mri_data),
                'seg': torch.from_numpy(seg)}
