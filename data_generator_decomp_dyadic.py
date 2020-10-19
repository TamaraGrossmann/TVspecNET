# ---------------------------------------------------------------------------------------------------------------------------
# @article{grossmann2020,
#    title={Deeply Learned Spectral Total Variation Decomposition},
#    author={Tamara G. Grossmann and Yury Korolev and Guy Gilboa and Carola-Bibiane Schönlieb},
#    year={2020},
#    eprint={2006.10004},
#    archivePrefix={arXiv},
#    primaryClass={cs.CV}
#  }
#
# Author: Tamara G. Grossmann (2020), tg410@cam.ac.uk
# This code was amended from Kai Zhang (08/2018) https://github.com/cszn/DnCNN
#
# Code for the data generator of training data from .mat data type images. Each input image is assumed to be in an individual 
# folder in which 50 ground truth decomposed bands are stored. The GT bands will be combined to 6 dyadic bands in this 
# implementation. 
# PyTorch Version 1.1.0 used for this implementation. 
# ----------------------------------------------------------------------------------------------------------------------------
import os, glob
# import cv2
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
import torch

class DecompositionDataset(Dataset):
    # Generation of ground truth decomposed dyadic bands
    def __init__(self, xs, xs_info):
        self.xs = xs # input images
        self.xs_info = xs_info # [xs_dir,mode_aug]

    def __getitem__(self, index):
        batch_x = self.xs[index]
        xs_index_info = self.xs_info[index]

        bands_list = sorted(glob.glob(xs_index_info[0] + '/filtered_image*_band*.mat')) 
        
        batch_y = []
        # generate the dyadic bands (assuming 50 GT bands)
        for i in range(1,6):
            batch_temp=[]
            for b in range((2**(i-1)-1),2**(i)-1):
            # decomposed bands
                batch = sio.loadmat(bands_list[b])
                batch = batch['f_H']
                batch_temp.append(batch)
            batch_temp = np.sum(np.array(batch_temp,dtype='float32'),0)
            batch_temp = data_aug(batch_temp,mode=xs_index_info[1].astype('int'))
            batch_y.append(batch_temp)
        batch_y = np.array(batch_y, dtype='float32')
        batch_y = torch.from_numpy(batch_y)
        
        return batch_x, batch_y
    
    def __len__(self):
        return self.xs.size(0)
            
        
# Data augmentation        
def data_aug(img, mode=0):
    if mode == 0: # no augmentation
        return img
    elif mode == 1:
        return np.rot90(img,k=2) # rotation by 180°
    elif mode == 2:
        return np.rot90(img) # rotation by 90°


def datagenerator(data_dir='data/train', verbose=False):
    # generate input images from a dataset
    file_list = glob.glob(data_dir+'/train*')  # get name list of all folders containing training images    
    data = []
    data_info = []
    # load images and apply data augmentation
    for f in range(len(file_list)):
        img_dir = os.path.join(file_list[f],'image.mat')
        img = sio.loadmat(img_dir)
        img = img['img']
        mode_aug=np.random.randint(0,3) # choose augmentation at random
        img = data_aug(img, mode=mode_aug)
        data.append(img)
        data_info.append([file_list[f],mode_aug]) # save augmentation method to apply to ground truth bands
    data = np.array(data, dtype='float32')
    data = np.expand_dims(data, axis=1)
    data_info = np.array(data_info, dtype='str')
    print('^_^-training data finished-^_^')
    return data, data_info
                                        

if __name__ == '__main__': 
    data = datagenerator(data_dir='test2')
