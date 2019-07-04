import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import nibabel as nib
import numpy as np
import glob

class MRIDataset(Dataset):
    """
    MRI Data set
    """
    
    def __init__(self, img_dir, idc, num_slices=155, transform_state=True):
        """
        Args:
            img_dir (string): path to data set directories
            idc (list of int): indice list
            num_slices - default (int): number of slices in MRI Volume
            transform - default (callable): for transformation to tensor 
        """
        
        self.idc = idc
        self.img_dir = img_dir
        self.nii_dir = np.asarray(glob.glob(self.img_dir + '/*'))[self.idc]
        self.nii_per_dir = len(glob.glob(self.nii_dir[0] + '/*.npy'))
        self.num_slices = num_slices
        self.transform_state = transform_state
                    
    def __len__(self):
        return len(self.nii_dir) * self.num_slices
    
    def __get_img__(self, img_name):
        """
        Returns the segmentation image
        """
        img_data = np.load(img_name, mmap_mode='r')
        return img_data
        
    def __transform__(self, sample):
        zscore = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0., 0., 0., 0.), 
                                                         (1., 1., 1., 1.))])
        seg = torch.from_numpy(sample['seg']).type(torch.LongTensor)
        # mri_data shape: CxHxW; new shape: HxWxC
        mri_data = np.asarray(sample['mri_data']).transpose((1,2,0))        
        mri_data = zscore(mri_data)
        path = sample['subject_slice_path']
        return {'mri_data': mri_data, 
                'seg': seg,
                'subject_slice_path': path}
    
    def __getitem__(self, idx):
        """
        supports the indexing of MRIDataset
        """
        # indices to subject idx
        sub_idx = idx // self.num_slices
        slice_idx = idx % len(self.nii_dir)
        path_nii = glob.glob(self.nii_dir[sub_idx] + '/*.npy')
        mri_data = []
        for path in path_nii:
            if path[-7:] == 'seg.npy':
                seg = self.__get_img__(path)[:,:,slice_idx]
            else:
                mri_data.append(self.__get_img__(path)[:,:,slice_idx])
        # return also path to subject directory for saving the output 
        # to the specific path
        sample = {'mri_data': np.asarray(mri_data), 'seg': np.asarray(seg),
                  'subject_slice_path': self.nii_dir[sub_idx] + str(slice_idx)}
        if self.transform_state:
            sample = self.__transform__(sample)
        return sample
