import nibabel as nib
import numpy as np
import glob

def add_path(splits):
    path = ''
    for sp in splits:
        path += sp + '/'
    return path
def load_save_nii_np(nii_filenames):  
    print(len(nii_filenames))              
    for i,name in enumerate(nii_filenames):
        image = nib.load(name)
        fdata = np.array(image.get_fdata().astype(np.float32))
        filename = str(add_path(name[:-7].split('/')[:-1]) + name[:-7].split('/')[-1] + '.npy')
        np.save(filename, fdata)
        del fdata
        del image
        print(i, filename, flush=True)

if __name__ == "__main__":
    
    target = '/home/hilkert/Dokumente/Uni/DeepVision/Project/05Data/BraTS/MICCAI_BraTS_2018_Data_Training/HGG'
    nii_dirs = glob.glob(target + '/*')
    nii_filenames = []
    for des in nii_dirs:
        nii_dirs = glob.glob(des + '/*')
        for nii in nii_dirs:
            nii_filenames.append(nii)

    load_save_nii_np(nii_filenames)

    print('Done', flush=True)

    
