# Author: Firat Ozdemir, May 2022, firat.ozdemir@datascience.ch
## Utility scripts 

import datetime
import glob
import os
import pickle
import shutil
import time
import numpy as np
import h5py
import sklearn.metrics
import skimage.metrics
import tensorflow as tf

def load_model_hyperparameters(task_str, datasets_parent_dir='.'):
    if task_str in ['msfd_lv128,li', 'msfd_ss32,ms', 'msfd_ss64,ms', 'msfd_ss128,ms']:
        hyperparams = ImageTranslationSetups
    elif task_str in ["swfd_lv128,sc", "swfd_ss32,sc", "swfd_ss64,sc", "swfd_ss128,sc", "scd_ss32,vc", "scd_lv128,vc", "scd_ss64,vc", "scd_lv128,li", "swfd_lv128,li", "scd_lv256,ms", "scd_ss128,vc", 'swfd_ss32,ms', 'swfd_ss64,ms', 'swfd_ss128,ms', 'scd_ss128,ms', 'scd_ss64,ms', 'scd_ss32,ms']: ## other image translation
        hyperparams = ImageTranslationSetups
    elif task_str in ["seg_ss32,vc", "seg_ss128,vc",  "seg_ss64,vc", "seg_lv128,li", "seg_lv128,vc", "seg_vc,vc", 'seg_swfd_ss32,ms', 'seg_swfd_ss64,ms', 'seg_swfd_ss128,ms', 'seg_swfd_ss32,sc', 'seg_swfd_ss64,sc', 'seg_swfd_ss128,sc', 'seg_swfd_sc,sc', 'seg_swfd_ms,ms', 'seg_msfd_ss32,ms', 'seg_msfd_ss64,ms', 'seg_msfd_ss128,ms', 'seg_ss32,ms', 'seg_ss64,ms', 'seg_ss128,ms', 'seg_swfd_lv128,ms', 'seg_swfd_lv128,sc', 'seg_msfd_lv128,ms']: ## segmentation
        hyperparams = ImageSegmentationSetups
    else:
        raise NotImplementedError(f'Unexpected task_str: {task_str}')
    hpobj = hyperparams(task_str=task_str, datasets_parent_dir=datasets_parent_dir)
    return hpobj


class ImageTranslationSetups:
    '''Helper class to fix certain groups of hyperparameters across different paired image translation task experiments'''
    def __init__(self, task_str, datasets_parent_dir='.'):
        self.task_str = str(task_str).lower()
        self.datasets_parent_dir = datasets_parent_dir
        self.setup_hyperparameters()

    def split_data(self, data_str):
        if not os.path.isfile(self.fname_h5):
            print(f"Dataset file {self.fname_h5} not found, will not setup train/val/test indices.")
            self.inds_train, self.inds_val, self.inds_test = None, None, None
            return
        if str(data_str).lower() == 'swfd':
            with h5py.File(self.fname_h5, 'r') as fh:
                pIDs = fh['patientID'][()]
            pID_unique = np.unique(pIDs)
            assert len(pID_unique) == 14
            pID_train = pID_unique[:8] # 8 subjects, 22416 images
            pID_val = pID_unique[8:9] # 1 subject, 2802 images
            pID_test = pID_unique[9:14] # 5 subjects, 14010 images
            self.inds_train = np.sort(np.concatenate([np.where(pIDs==i)[0] for i in pID_train], axis=0))
            self.inds_val = np.sort(np.concatenate([np.where(pIDs==i)[0] for i in pID_val], axis=0))
            self.inds_test = np.sort(np.concatenate([np.where(pIDs==i)[0] for i in pID_test], axis=0))      
        elif str(data_str).lower() == 'scd':
            def get_num_samples(fname_h5):
                with h5py.File(fname_h5, 'r') as fh:
                    return fh[self.out_key].shape[0]

            num_samples = get_num_samples(self.fname_h5) #20k
            inds = np.arange(num_samples)
            
            # self.prng.shuffle(inds)
            ratio_train_val_test = [0.7, 0.05, 0.25] # 14k, 1k, 5k
            assert np.sum(ratio_train_val_test) == 1.0
            inds_train_val_test = np.array(np.cumsum(ratio_train_val_test) * num_samples, dtype=int)
            self.inds_train = np.sort(inds[:inds_train_val_test[0]])
            self.inds_val = np.sort(inds[inds_train_val_test[0]:inds_train_val_test[1]])
            self.inds_test = np.sort(inds[inds_train_val_test[1]:inds_train_val_test[2]])
        elif str(data_str).lower() == 'msfd':
            with h5py.File(self.fname_h5, 'r') as fh:
                pIDs = fh['patientID'][()]
            pID_unique = np.unique(pIDs)
            assert len(pID_unique) == 9
            pID_train = pID_unique[:5] # 5 subjects, 14000x6 images
            pID_val = pID_unique[5:6] # 1 subject, 2800x6 images
            pID_test = pID_unique[6:] # 3 subjects, 8400x6 images
            self.inds_train = np.sort(np.concatenate([np.where(pIDs==i)[0] for i in pID_train], axis=0))
            self.inds_val = np.sort(np.concatenate([np.where(pIDs==i)[0] for i in pID_val], axis=0))
            self.inds_test = np.sort(np.concatenate([np.where(pIDs==i)[0] for i in pID_test], axis=0))      
    
    def setup_hyperparameters(self):
        self.model_name_base_str = "modified_unet_%s_MSE"%self.task_str
        if self.task_str == 'swfd_lv128,li':
            '''SWFD, limited view. 128 elem. linear array to 256 elem multisegment array'''
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SWFD_multisegment_RawBP.h5") 
            self.in_key = 'linear_BP' ## central 128 elements of multisegment array constitutes a linear array
            self.out_key = 'ms_BP' ## multisegment array has 256 elements.
            self.split_data(data_str='swfd')
            self.dataset_base_str = "SWFD_multisegment"

        elif self.task_str == 'swfd_lv128,sc':
            '''SWFD, limited view. 128 elem. semi-circle array to 256 elem semi-circle array'''
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SWFD_semicircle_RawBP.h5")
            self.in_key = 'sc,lv128_BP' ## 128 consecutive elements of semi-circle array
            self.out_key = 'sc_BP' ## all 256 elements of the semi-circle array
            self.split_data(data_str='swfd')
            self.dataset_base_str = "SWFD_semicircle"

        elif self.task_str == 'swfd_ss128,sc':
            '''SWFD, sparse view. 128 elem. semi-circle array to 256 elem semi-circle array'''
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SWFD_semicircle_RawBP.h5")
            self.in_key = 'sc,ss128_BP' ## uniformly distributed sparse 128 elements of the semi-circle array
            self.out_key = 'sc_BP' ## all 256 elements of the semi-circle array
            self.split_data(data_str='swfd')
            self.dataset_base_str = "SWFD_semicircle"

        elif self.task_str == 'swfd_ss64,sc':
            '''SWFD, sparse view. 64 elem. semi-circle array to 256 elem semi-circle array'''
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SWFD_semicircle_RawBP.h5")
            self.in_key = 'sc,ss64_BP' ## uniformly distributed sparse 64 elements of the semi-circle array
            self.out_key = 'sc_BP' ## all 256 elements of the semi-circle array
            self.split_data(data_str='swfd')
            self.dataset_base_str = "SWFD_semicircle"

        elif self.task_str == 'swfd_ss32,sc':
            '''SWFD, sparse view. 32 elem. semi-circle array to 256 elem semi-circle array'''
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SWFD_semicircle_RawBP.h5")
            self.in_key = 'sc,ss32_BP' ## uniformly distributed sparse 32 elements of the semi-circle array
            self.out_key = 'sc_BP' ## all 256 elements of the semi-circle array
            self.split_data(data_str='swfd')
            self.dataset_base_str = "SWFD_semicircle"

        elif self.task_str == 'swfd_ss128,ms':
            '''SWFD, sparse view. 128 elem. multisegment array to 256 elem multisegment array'''
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SWFD_multisegment_ss_RawBP.h5")
            self.in_key = 'ms,ss128_BP' ## uniformly distributed sparse 128 elements of the multisegment array
            self.out_key = 'ms_BP' ## all 256 elements of the multisegment array
            self.split_data(data_str='swfd')
            self.dataset_base_str = "SWFD_multisegment"

        elif self.task_str == 'swfd_ss64,ms':
            '''SWFD, sparse view. 64 elem. multisegment array to 256 elem multisegment array'''
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SWFD_multisegment_ss_RawBP.h5")
            self.in_key = 'ms,ss64_BP' ## uniformly distributed sparse 64 elements of the multisegment array
            self.out_key = 'ms_BP' ## all 256 elements of the multisegment array
            self.split_data(data_str='swfd')
            self.dataset_base_str = "SWFD_multisegment"

        elif self.task_str == 'swfd_ss32,ms':
            '''SWFD, sparse view. 32 elem. multisegment array to 256 elem multisegment array'''
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SWFD_multisegment_ss_RawBP.h5")
            self.in_key = 'ms,ss32_BP' ## uniformly distributed sparse 32 elements of the multisegment array
            self.out_key = 'ms_BP' ## all 256 elements of the multisegment array
            self.split_data(data_str='swfd')
            self.dataset_base_str = "SWFD_multisegment"


        ### Simulated dataset exps below
        elif self.task_str == 'scd_ss32,vc':
            '''SCD, sparse view. 32 elem. circle array to 1024 elem circle array'''
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SCD_RawBP.h5")
            self.in_key = 'vc,ss32_BP' ## uniformly distributed sparse 32 elements of the circle array
            self.out_key = 'vc_BP' ## all 1024 elements of the semi-circle array
            self.split_data(data_str='scd')
            self.dataset_base_str = "SCD_circle"

        elif self.task_str == 'scd_ss64,vc':
            '''SCD, sparse view. 64 elem. circle array to 1024 elem circle array'''
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SCD_RawBP.h5")
            self.in_key = 'vc,ss64_BP' ## uniformly distributed sparse 64 elements of the circle array
            self.out_key = 'vc_BP' ## all 1024 elements of the semi-circle array
            self.split_data(data_str='scd')
            self.dataset_base_str = "SCD_circle"

        elif self.task_str == 'scd_ss128,vc':
            '''SCD, sparse view. 128 elem. circle array to 1024 elem circle array'''
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SCD_RawBP.h5")
            self.in_key = 'vc,ss128_BP' ## uniformly distributed sparse 128 elements of the circle array
            self.out_key = 'vc_BP' ## all 1024 elements of the semi-circle array
            self.split_data(data_str='scd')
            self.dataset_base_str = "SCD_circle"

        elif self.task_str == 'scd_lv128,vc':
            '''SCD, limited view. 128 elem. circle array to 1024 elem circle array'''
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SCD_RawBP.h5")
            self.in_key = 'vc,lv128_BP' ## 128 consecutive elements of circle array
            self.out_key = 'vc_BP' ## all 1024 elements of the semi-circle array
            self.split_data(data_str='scd')
            self.dataset_base_str = "SCD_circle"

        elif self.task_str == 'scd_lv128,li':
            '''SCD, limited view. 128 elem. linear array to 1024 elem circle array'''
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SCD_RawBP.h5")
            self.in_key = 'linear_BP' ## 128 consecutive elements of linear array
            self.out_key = 'vc_BP' ## all 1024 elements of the semi-circle array
            self.split_data(data_str='scd')
            self.dataset_base_str = "SCD_linear"

        elif self.task_str == 'scd_lv256,ms':
            '''SCD, limited view. 256 elem. multisegment array to 1024 elem circle array'''
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SCD_RawBP.h5")
            self.in_key = 'ms_BP' ## 256 consecutive elements of linear array
            self.out_key = 'vc_BP' ## all 1024 elements of the semi-circle array
            self.split_data(data_str='scd')
            self.dataset_base_str = "SCD_multisegment"

        elif self.task_str == 'scd_ss128,ms':
            '''SCD, sparse view. 128 elem. circle array to 1024 elem circle array'''
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SCD_multisegment_ss_RawBP.h5")
            self.in_key = 'ms,ss128_BP'  ## uniformly distributed sparse 128 elements of the multisegment array
            self.out_key = 'vc_BP' ## all 1024 elements of the semi-circle array
            self.split_data(data_str='scd')
            self.dataset_base_str = "SCD_multisegment"

        elif self.task_str == 'scd_ss64,ms':
            '''SCD, sparse view. 64 elem. circle array to 1024 elem circle array'''
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SCD_multisegment_ss_RawBP.h5")
            self.in_key = 'ms,ss64_BP'  ## uniformly distributed sparse 64 elements of the multisegment array
            self.out_key = 'vc_BP' ## all 1024 elements of the semi-circle array
            self.split_data(data_str='scd')
            self.dataset_base_str = "SCD_multisegment"

        elif self.task_str == 'scd_ss32,ms':
            '''SCD, sparse view. 32 elem. circle array to 1024 elem circle array'''
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SCD_multisegment_ss_RawBP.h5")
            self.in_key = 'ms,ss32_BP'  ## uniformly distributed sparse 32 elements of the multisegment array
            self.out_key = 'vc_BP' ## all 1024 elements of the semi-circle array
            self.split_data(data_str='scd')
            self.dataset_base_str = "SCD_multisegment"

        ## MSFD below
        elif self.task_str == 'msfd_lv128,li':
            '''MSFD, limited view. 128 elem. linear array to 256 elem multisegment array where all wavelengths are mapped onto corresponding wavelength images.'''
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "MSFD_multisegment_RawBP.h5")
            self.wavelengths = [700, 730, 760, 780, 800, 850] #[nm]
            self.in_key = ['linear_BP_w%d'%w for w in self.wavelengths]  ## central 128 elements of multisegment array constitutes a linear array
            self.out_key = ['ms_BP_w%d'%w for w in self.wavelengths] ## multisegment array has 256 elements.
            self.split_data(data_str='msfd')
            self.dataset_base_str = "MSFD_multisegment"

        elif self.task_str == 'msfd_ss32,ms':
            '''MSFD, sparse view. 32 elem. multisegment array to 256 elem multisegment array where all wavelengths are mapped onto corresponding wavelength images.'''
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "MSFD_multisegment_ss_RawBP.h5")
            self.wavelengths = [700, 730, 760, 780, 800, 850] #[nm]
            self.in_key = ['ms,ss32_BP_w%d'%w for w in self.wavelengths]  ## uniformly distributed sparse 32 elements of the multisegment array
            self.out_key = ['ms_BP_w%d'%w for w in self.wavelengths] ## multisegment array has 256 elements.
            self.split_data(data_str='msfd')
            self.dataset_base_str = "MSFD_multisegment"

        elif self.task_str == 'msfd_ss64,ms':
            '''MSFD, sparse view. 64 elem. multisegment array to 256 elem multisegment array where all wavelengths are mapped onto corresponding wavelength images.'''
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "MSFD_multisegment_ss_RawBP.h5")
            self.wavelengths = [700, 730, 760, 780, 800, 850] #[nm]
            self.in_key = ['ms,ss64_BP_w%d'%w for w in self.wavelengths]  ## uniformly distributed sparse 64 elements of the multisegment array
            self.out_key = ['ms_BP_w%d'%w for w in self.wavelengths] ## multisegment array has 256 elements.
            self.split_data(data_str='msfd')
            self.dataset_base_str = "MSFD_multisegment"

        elif self.task_str == 'msfd_ss128,ms':
            '''MSFD, sparse view. 128 elem. multisegment array to 256 elem multisegment array where all wavelengths are mapped onto corresponding wavelength images.'''
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "MSFD_multisegment_ss_RawBP.h5")
            self.wavelengths = [700, 730, 760, 780, 800, 850] #[nm]
            self.in_key = ['ms,ss128_BP_w%d'%w for w in self.wavelengths]  ## uniformly distributed sparse 128 elements of the multisegment array
            self.out_key = ['ms_BP_w%d'%w for w in self.wavelengths] ## multisegment array has 256 elements.
            self.split_data(data_str='msfd')
            self.dataset_base_str = "MSFD_multisegment"

        else:
            raise AssertionError(f'Unknown task_str: {self.task_str}')


class ImageSegmentationSetups:
    '''Helper class to fix certain groups of hyperparameters across different supervised segmentation task experiments'''
    def __init__(self, task_str, datasets_parent_dir='.'):
        self.task_str = str(task_str).lower()
        self.datasets_parent_dir = datasets_parent_dir
        self.setup_hyperparameters()

    def split_data(self, data_str):
        if not os.path.isfile(self.fname_h5):
            print(f"Dataset file {self.fname_h5} not found, will not setup train/val/test indices.")
            self.inds_train, self.inds_val, self.inds_test = None, None, None
            return
        if str(data_str).lower() == 'scd':
            self.labels = {'bg': 0, 'vessel': 1, 'skincurve': 2}
            def count_label_freq(inds, fname_in, k_gt='labels', recount=False):
                inds = np.sort(inds)
                if not recount:
                    l_label_freq = [0.9715059215000698, 0.015896890912737164, 0.01259718758719308]
                else:
                    l_label_freq = []
                    with h5py.File(fname_in, 'r') as fh:
                        for k in self.labels.keys():
                            l_label_freq.append(np.sum(fh[k_gt][inds,...] == self.labels[k]) / np.prod(fh[k_gt][inds,...].shape)) # values in [0, 1]
                return l_label_freq

            def get_num_samples(fname_h5):
                with h5py.File(fname_h5, 'r') as fh:
                    return fh[self.out_key].shape[0]

            num_samples = get_num_samples(self.fname_h5) #20k
            inds = np.arange(num_samples)
            
            # self.prng.shuffle(inds)
            ratio_train_val_test = [0.7, 0.05, 0.25] # 14k, 1k, 5k
            assert np.sum(ratio_train_val_test) == 1.0
            inds_train_val_test = np.array(np.cumsum(ratio_train_val_test) * num_samples, dtype=int)
            self.inds_train = np.sort(inds[:inds_train_val_test[0]])
            self.inds_val = np.sort(inds[inds_train_val_test[0]:inds_train_val_test[1]])
            self.inds_test = np.sort(inds[inds_train_val_test[1]:inds_train_val_test[2]])

            l_label_freq = count_label_freq(inds=self.inds_train, fname_in=self.fname_h5, k_gt=self.out_key)
            self.label_inv_freq = None
            if self.use_inv_freq_labels:
                self.label_inv_freq = [1/c for c in l_label_freq] # N*1/freq
                self.label_inv_freq = np.asarray(self.label_inv_freq)
                N = np.sum(self.label_inv_freq)
                self.label_inv_freq = np.asarray(self.label_inv_freq) / N # normalize weights to [0, 1]
        elif 'mini' in str(data_str).lower():
            self.labels = {'bg': 0, 'vessel': 1}
            def count_label_freq(inds, fname_in, k_gt='labels', recount=False):
                inds = np.sort(inds)
                l_label_freq = []
                with h5py.File(fname_in, 'r') as fh:
                    for k in self.labels.keys():
                        l_label_freq.append(np.sum(fh[k_gt][inds,...] == self.labels[k]) / np.prod(fh[k_gt][inds,...].shape)) # values in [0, 1]
                return l_label_freq
            
            def get_num_samples(fname_h5):
                with h5py.File(fname_h5, 'r') as fh:
                    if 'msfd' in self.task_str: # msfd-mini
                        out_key = self.out_key[0] 
                    else:
                        out_key = self.out_key
                    return fh[out_key].shape[0]
                    
            num_samples = get_num_samples(self.fname_h5) #100
            inds = np.arange(num_samples)
            
            # self.prng.shuffle(inds)
            ratio_train_val_test = [0.75, 0.05, 0.2] # 75, 5, 20
            assert np.sum(ratio_train_val_test) == 1.0
            inds_train_val_test = np.array(np.cumsum(ratio_train_val_test) * num_samples, dtype=int)
            self.inds_train = np.sort(inds[:inds_train_val_test[0]])
            self.inds_val = np.sort(inds[inds_train_val_test[0]:inds_train_val_test[1]])
            self.inds_test = np.sort(inds[inds_train_val_test[1]:inds_train_val_test[2]])

            if 'msfd' in self.task_str: # msfd-mini
                l_label_freq = count_label_freq(inds=self.inds_train, fname_in=self.fname_h5, k_gt=self.out_key[0])
            else:
                l_label_freq = count_label_freq(inds=self.inds_train, fname_in=self.fname_h5, k_gt=self.out_key)
            self.label_inv_freq = None
            if self.use_inv_freq_labels:
                self.label_inv_freq = [1/c for c in l_label_freq] # N*1/freq
                self.label_inv_freq = np.asarray(self.label_inv_freq)
                N = np.sum(self.label_inv_freq)
                self.label_inv_freq = np.asarray(self.label_inv_freq) / N # normalize weights to [0, 1]
    
    def setup_hyperparameters(self):
        self.model_name_base_str = "modified_unet_%s_xent"%self.task_str
        self.use_inv_freq_labels = True
        if self.task_str == 'seg_lv128,li':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SCD_RawBP.h5")
            self.in_key = 'linear_BP' ## central 128 elements of multisegment array constitutes a linear array
            self.out_key = 'labels' ## annotation map
            self.split_data(data_str='scd')
            self.dataset_base_str = "SCD"

        elif self.task_str == 'seg_lv128,vc':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SCD_RawBP.h5")
            self.in_key = 'vc,lv128_BP' ## central 128 elements of circle array 
            self.out_key = 'labels' ## annotation map
            self.split_data(data_str='scd')
            self.dataset_base_str = "SCD"

        elif self.task_str == 'seg_ss128,vc':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SCD_RawBP.h5")
            self.in_key = 'vc,ss128_BP' ## uniformly distributed sparse 128 elements of the circle array
            self.out_key = 'labels' ## annotation map
            self.split_data(data_str='scd')
            self.dataset_base_str = "SCD"

        elif self.task_str == 'seg_ss64,vc':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SCD_RawBP.h5")
            self.in_key = 'vc,ss64_BP' ## uniformly distributed sparse 64 elements of the circle array
            self.out_key = 'labels' ## annotation map
            self.split_data(data_str='scd')
            self.dataset_base_str = "SCD"

        elif self.task_str == 'seg_ss32,vc':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SCD_RawBP.h5")
            self.in_key = 'vc,ss32_BP' ## uniformly distributed sparse 32 elements of the circle array
            self.out_key = 'labels' ## annotation map
            self.split_data(data_str='scd')
            self.dataset_base_str = "SCD"
            
        elif self.task_str == 'seg_vc,vc':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SCD_RawBP.h5")
            self.in_key = 'vc_BP' ## uniformly distributed sparse 32 elements of the circle array
            self.out_key = 'labels' ## annotation map
            self.split_data(data_str='scd')
            self.dataset_base_str = "SCD"

        elif self.task_str == 'seg_ss32,ms':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SCD_multisegment_ss_RawBP.h5")
            self.in_key = 'ms,ss32_BP' ## uniformly distributed sparse 32 elements of the multisegment array
            self.out_key = 'labels' ## annotation map
            self.split_data(data_str='scd')
            self.dataset_base_str = "SCD"

        elif self.task_str == 'seg_ss64,ms':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SCD_multisegment_ss_RawBP.h5")
            self.in_key = 'ms,ss64_BP' ## uniformly distributed sparse 64 elements of the multisegment array
            self.out_key = 'labels' ## annotation map
            self.split_data(data_str='scd')
            self.dataset_base_str = "SCD"

        elif self.task_str == 'seg_ss128,ms':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SCD_multisegment_ss_RawBP.h5")
            self.in_key = 'ms,ss128_BP' ## uniformly distributed sparse 128 elements of the multisegment array
            self.out_key = 'labels' ## annotation map
            self.split_data(data_str='scd')
            self.dataset_base_str = "SCD"

        elif self.task_str == 'seg_swfd_ss32,ms':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SWFD_multisegment_RawBP-mini.h5")
            self.in_key = 'ms,ss32_BP' ## uniformly distributed sparse 32 elements of the multisegment array
            self.out_key = 'labels' ## annotation map
            self.split_data(data_str='mini')
            self.dataset_base_str = "SWFD_multisegment-mini"

        elif self.task_str == 'seg_swfd_ss64,ms':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SWFD_multisegment_RawBP-mini.h5")
            self.in_key = 'ms,ss64_BP' ## uniformly distributed sparse 64 elements of the multisegment array
            self.out_key = 'labels' ## annotation map
            self.split_data(data_str='mini')
            self.dataset_base_str = "SWFD_multisegment-mini"

        elif self.task_str == 'seg_swfd_ss128,ms':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SWFD_multisegment_RawBP-mini.h5")
            self.in_key = 'ms,ss128_BP' ## uniformly distributed sparse 128 elements of the multisegment array
            self.out_key = 'labels' ## annotation map
            self.split_data(data_str='mini')
            self.dataset_base_str = "SWFD_multisegment-mini"

        elif self.task_str == 'seg_swfd_lv128,ms':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SWFD_multisegment_RawBP-mini.h5")
            self.in_key = 'linear_BP' ## limited view 128 elements of the multisegment array
            self.out_key = 'labels' ## annotation map
            self.split_data(data_str='mini')
            self.dataset_base_str = "SWFD_multisegment-mini"

        elif self.task_str == 'seg_swfd_lv128,sc':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SWFD_semicircle_RawBP-mini.h5")
            self.in_key = 'sc,lv128_BP' ## limited view 128 elements of the semi circle array
            self.out_key = 'labels' ## annotation map
            self.split_data(data_str='mini')
            self.dataset_base_str = "SWFD_semicircle-mini"

        elif self.task_str == 'seg_swfd_ss32,sc':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SWFD_semicircle_RawBP-mini.h5")
            self.in_key = 'sc,ss32_BP' ## uniformly distributed sparse 32 elements of the semi-circle array
            self.out_key = 'labels' ## annotation map
            self.split_data(data_str='mini')
            self.dataset_base_str = "SWFD_semicircle-mini"

        elif self.task_str == 'seg_swfd_ss64,sc':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SWFD_semicircle_RawBP-mini.h5")
            self.in_key = 'sc,ss64_BP' ## uniformly distributed sparse 64 elements of the semi-circle array
            self.out_key = 'labels' ## annotation map
            self.split_data(data_str='mini')
            self.dataset_base_str = "SWFD_semicircle-mini"

        elif self.task_str == 'seg_swfd_ss128,sc':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SWFD_semicircle_RawBP-mini.h5")
            self.in_key = 'sc,ss128_BP' ## uniformly distributed sparse 128 elements of the semi-circle array
            self.out_key = 'labels' ## annotation map
            self.split_data(data_str='mini')
            self.dataset_base_str = "SWFD_semicircle-mini"

        elif self.task_str == 'seg_swfd_sc,sc':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SWFD_semicircle_RawBP-mini.h5")
            self.in_key = 'sc_BP' ## uniformly distributed sparse 128 elements of the semi-circle array
            self.out_key = 'labels' ## annotation map
            self.split_data(data_str='mini')
            self.dataset_base_str = "SWFD_semicircle-mini"

        elif self.task_str == 'seg_swfd_ms,ms':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "SWFD_multisegment_RawBP-mini.h5")
            self.in_key = 'ms_BP' ## uniformly distributed sparse 128 elements of the semi-circle array
            self.out_key = 'labels' ## annotation map
            self.split_data(data_str='mini')
            self.dataset_base_str = "SWFD_multisegment-mini"

        elif self.task_str == 'seg_msfd_ss32,ms':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "MSFD_multisegment_RawBP-mini.h5")
            self.wavelengths = [700, 730, 760, 780, 800, 850] #[nm]
            self.in_key = ['ms,ss32_BP_w%d'%w for w in self.wavelengths]  ## uniformly distributed sparse 32 elements of the multisegment array
            self.out_key = ['labels' for w in self.wavelengths] ## annotation map
            self.split_data(data_str='mini')
            self.dataset_base_str = "MSFD_multisegment-mini"
            
        elif self.task_str == 'seg_msfd_ss64,ms':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "MSFD_multisegment_RawBP-mini.h5")
            self.wavelengths = [700, 730, 760, 780, 800, 850] #[nm]
            self.in_key = ['ms,ss64_BP_w%d'%w for w in self.wavelengths]  ## uniformly distributed sparse 64 elements of the multisegment array
            self.out_key = ['labels' for w in self.wavelengths] ## annotation map
            self.split_data(data_str='mini')
            self.dataset_base_str = "MSFD_multisegment-mini"
            
        elif self.task_str == 'seg_msfd_ss128,ms':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "MSFD_multisegment_RawBP-mini.h5")
            self.wavelengths = [700, 730, 760, 780, 800, 850] #[nm]
            self.in_key = ['ms,ss128_BP_w%d'%w for w in self.wavelengths]  ## uniformly distributed sparse 128 elements of the multisegment array
            self.out_key = ['labels' for w in self.wavelengths] ## annotation map
            self.split_data(data_str='mini')
            self.dataset_base_str = "MSFD_multisegment-mini"

        elif self.task_str == 'seg_msfd_lv128,ms':
            self.fname_h5 = os.path.join(self.datasets_parent_dir, "MSFD_multisegment_RawBP-mini.h5")
            self.wavelengths = [700, 730, 760, 780, 800, 850] #[nm]
            self.in_key = ['linear_BP_w%d'%w for w in self.wavelengths]  ## limited view 128 elements of the multisegment array
            self.out_key = ['labels' for w in self.wavelengths] ## annotation map
            self.split_data(data_str='mini')
            self.dataset_base_str = "MSFD_multisegment-mini"

        else:
            raise AssertionError('Unknown task_str.')

class Metrics_Translation:
    def __init__(self) -> None:
        self.mae_fn = lambda ypred, ytrue: sklearn.metrics.mean_absolute_error(ytrue, ypred)
        self.rmse_fn = lambda ypred, ytrue: np.sqrt(skimage.metrics.mean_squared_error(ytrue, ypred))
        self.ssim_fn = lambda ypred, ytrue: skimage.metrics.structural_similarity(ytrue, ypred, data_range=ypred.max()-ypred.min())
        self.psnr_fn = lambda ypred, ytrue: skimage.metrics.peak_signal_noise_ratio(ytrue, ypred, data_range=ypred.max()-ypred.min())
    
    def compute(self, ypred, ytrue):
        if np.shape(ypred)[0] == 256 and np.shape(ytrue)[0] == 256:
            d_res_shp = ()
        elif np.shape(ypred) != np.shape(ytrue):
            raise AssertionError(f'Shapes of ypred ({np.shape(ypred)}) and ytrue ({np.shape(ytrue)}) must match!')
        else:
            d_res_shp = (np.shape(ytrue)[0],)
        d_res = {'MAE': np.inf*np.ones(d_res_shp), 'RMSE': np.inf*np.ones(d_res_shp), 'SSIM': np.inf*np.ones(d_res_shp), 'PSNR': np.inf*np.ones(d_res_shp)}
        if len(d_res_shp) == 0:
            d_res['MAE'] = self.mae_fn(ypred, ytrue)
            d_res['RMSE'] = self.rmse_fn(ypred, ytrue)
            d_res['SSIM'] = self.ssim_fn(ypred, ytrue)
            d_res['PSNR'] = self.psnr_fn(ypred, ytrue)
        else:
            for i in range(len(d_res_shp)):
                d_res['MAE'][i] = self.mae_fn(ypred[i,...], ytrue[i,...])
                d_res['RMSE'][i] = self.rmse_fn(ypred[i,...], ytrue[i,...])
                d_res['SSIM'][i] = self.ssim_fn(ypred[i,...], ytrue[i,...])
                d_res['PSNR'][i] = self.psnr_fn(ypred[i,...], ytrue[i,...])
        return d_res

class Metrics_Segmentation:
    def __init__(self, num_classes) -> None:
        self.num_classes = num_classes
        self.precision_fn = lambda tp, fp: tp/(tp+fp)
        self.recall_fn = lambda tp, fn: tp/(tp+fn)
        self.f1_fn = lambda tp, fp, fn: 2*tp/(2*tp + fp + fn) # also dice
        self.IoU_fn = lambda tp, fp, fn: tp/(tp + fp + fn) # also Jaccard index


    def compute(self, ypred, ytrue):
        '''Expects inputs to be of shape [num_samples, 256, 256] or [256, 256]'''
        num_classes = self.num_classes

        if np.shape(ypred)[0] == 256 and np.shape(ytrue)[0] == 256:
            d_res_shp = (num_classes,)
            shp_Parray = (-1)
        elif np.shape(ypred) != np.shape(ytrue):
            raise AssertionError(f'Shapes of ypred ({np.shape(ypred)}) and ytrue ({np.shape(ytrue)}) must match!')
        else:
            d_res_shp = (np.shape(ytrue)[0], num_classes,)
            shp_Parray = (np.shape(ytrue)[0], -1)
        
        d_res = {'precision': np.inf*np.ones(d_res_shp), 'recall': np.inf*np.ones(d_res_shp), 'dice': np.inf*np.ones(d_res_shp), 'iou': np.inf*np.ones(d_res_shp), 'TP': np.inf*np.ones(d_res_shp) , 'FP': np.inf*np.ones(d_res_shp), 'TN': np.inf*np.ones(d_res_shp), 'FN': np.inf*np.ones(d_res_shp)}
        for lbl in range(num_classes):
            predP = np.reshape(ypred == lbl, shp_Parray) #shape: [bs, -1]
            trueP = np.reshape(ytrue == lbl, shp_Parray)
            TP = np.logical_and(predP, trueP)
            FP = np.logical_and(predP, ~trueP)
            TN = np.logical_and(~predP, ~trueP)
            FN = np.logical_and(~predP, trueP)
            TP = np.sum(TP, axis=-1)
            FP = np.sum(FP, axis=-1)
            TN = np.sum(TN, axis=-1)
            FN = np.sum(FN, axis=-1)

            precision = self.precision_fn(tp=TP, fp=FP)
            recall = self.recall_fn(tp=TP, fn=FN)
            dice = self.f1_fn(tp=TP, fp=FP, fn=FN)
            IoU = self.IoU_fn(tp=TP, fp=FP, fn=FN)

            d_res['precision'][..., lbl] = precision
            d_res['recall'][..., lbl] = recall
            d_res['dice'][..., lbl] = dice
            d_res['iou'][..., lbl] = IoU
            d_res['TP'][..., lbl] = TP
            d_res['FP'][..., lbl] = FP
            d_res['TN'][..., lbl] = TN
            d_res['FN'][..., lbl] = FN
        return d_res

class TrainingHelper:
    """class to contain certain variables to help continue training in case interrupted. Pickleble."""

    def __init__(self, fname_save):
        self.fname_save = fname_save
        self.l_generators = []  # will be populated with (name, generator_obj) tuple.
        self.l_gen_states = ([])  # populate with (name, state) tuple of each generator at each ckpt save using add_generator
        self.epoch_current = None
        self.best_val_loss = np.inf

    def add_generator(self, name, gen):
        self.l_generators.append((name, gen))

    def save(self, epoch_current, best_val_loss=None):
        """Function requires added generators to have a prng attribute."""
        self.l_gen_states = []
        for gen_tp in self.l_generators:
            name = gen_tp[0]
            gen = gen_tp[1]
            if not hasattr(gen, "prng"):
                raise AssertionError("TrainingHelper: Generator %s does not have a prng attribute!" % name)
            state = gen.prng.get_state()
            self.l_gen_states.append((name, state))
        d_content = {
            "l_gen_states": self.l_gen_states,
            "epoch_current": epoch_current,
            "best_val_loss": best_val_loss,
        }
        with open(self.fname_save, "wb") as fh:
            pickle.dump(d_content, fh)

    def load(self):
        if not os.path.isfile(self.fname_save):
            print("TrainingHelper: helper file %s does not exist. ")
            self.epoch_current = 0
            self.best_val_loss = np.inf
            return
        with open(self.fname_save, "rb") as fh:
            d_content = pickle.load(fh)
        self.epoch_current = d_content["epoch_current"]
        self.best_val_loss = d_content["best_val_loss"]
        for gen_tp in self.l_generators:
            k = gen_tp[0]
            gen = gen_tp[1]
            for gs_tp in d_content["l_gen_states"]:
                if gs_tp[0] == k:  # find the matching generator name
                    last_state = gs_tp[1]
                    gen.prng.set_state(last_state)
                    print("TrainingHelper: Loaded state for generator %s" % k)
                    break

    def save_training_script(self, file, logdir):
        """Saves this file to logdir. Ideally this is called after first successful training step, and only once."""
        current_script_name = os.path.basename(file)
        s_ = str(current_script_name).split(".")
        script_basename = "".join(s_[:-1])  # merge list of strings (in case multiple . exist in filename.)
        script_extension_name = s_[-1]
        src_file = os.path.join(os.getcwd(), current_script_name)
        if not os.path.isfile(src_file):
            src_file = os.path.join(os.getcwd(), 'src', current_script_name)
            if not os.path.isfile(src_file):
                print(f'Could not find training script under {src_file}. Will use {file}, make sure this is correct.')
                src_file = file
        out_file = os.path.join(logdir, current_script_name)
        if os.path.isfile(out_file):  # if outfile already exists, save another copy.
            date = datetime.datetime.now()
            datestr = date.strftime("%Y%m%d")
            out_file = os.path.join(logdir, script_basename + datestr + "." + script_extension_name)
            if os.path.isfile(out_file):  # if there is already a copy from that day, add time details also.
                datestr = date.strftime("%Y%m%d%H%M%S")
                out_file = os.path.join(logdir, script_basename + datestr + "." + script_extension_name)
        shutil.copyfile(src_file, out_file)

def get_ckptname(logdir, id):
    """Simple utility function to get checkpoint name from a directory of model checkpoints for a desired ckpt identifier."""
    try:
        fl = glob.glob(os.path.join(logdir, "*"))  # full paths of all files
        fn = [
            it for it in fl if id in os.path.basename(it)
        ]  # keep only files that match id in the basenames
        flb = [
            os.path.basename(it) for it in fn
        ]  # keep only basenames of matching abs path filenames
        split_chunks = [it[len(id) + 1 :].split(".") for it in flb] # for all matching filenames, split chunk after 'id' by "."
        if any(len(it) < 2 for it in split_chunks):
            # if any split has less than 2 items, chances are that id was an exact ID for a ckpt name and not a generic.
            ind = np.argmax([len(it) < 2 for it in split_chunks])
        else:
            ind = np.argmax(
                [int(it[len(id) + 1 :].split(".")[0]) for it in flb]
            )  # get highest ckpt ind of matching filenames
        fname = fn[ind]  # get abs filename of the matching index file
        fname = ".".join(
            fname.split(".")[:-1]
        )  # truncate extension of the file: e.g., /path/to/file/{id}-{ind}.{extension} -> /path/to/file/{id}-{ind}
        return fname
    except:
        print('Error: get_ckptname function failed to identify ckpt.')
        return None


class Manage_Pretrained_Models:
    def __init__(self):
        self.d_model_paths = {}
        self._build_lookup()
        self.list_task_str = ['msfd_lv128,li', 'swfd_lv128,li', 'swfd_lv128,sc', 'swfd_ss32,sc', 'swfd_ss64,sc', 'swfd_ss128,sc', 'scd_ss32,vc', 'scd_lv128,vc', 'scd_ss64,vc', 'scd_ss128,vc', 'scd_lv128,li', 'scd_lv256,ms', 'seg_ss32,vc', 'seg_ss128,vc', 'seg_ss64,vc', 'seg_lv128,li', 'seg_lv128,vc', 'seg_vc,vc', 'swfd_ss32,ms', 'swfd_ss64,ms', 'swfd_ss128,ms', 'seg_swfd_ss32,ms', 'seg_swfd_ss64,ms', 'seg_swfd_ss128,ms', 'seg_swfd_ss32,sc', 'seg_swfd_ss64,sc', 'seg_swfd_ss128,sc', 'seg_swfd_sc,sc', 'seg_swfd_ms,ms', 'seg_msfd_ss32,ms', 'seg_msfd_ss64,ms', 'seg_msfd_ss128,ms', 'scd_ss32,ms', 'scd_ss64,ms', 'scd_ss128,ms', 'seg_ss32,ms', 'seg_ss64,ms', 'seg_ss128,ms', 'seg_swfd_lv128,ms', 'seg_swfd_lv128,sc', 'seg_msfd_lv128,ms', 'msfd_ss32,ms', 'msfd_ss64,ms', 'msfd_ss128,ms']

    def _build_lookup(self):
        # MSFD image translation limited view 128 linear to multisegment BP 
        model_path = 'pretrained_models/msfd_lv128,li/serialized_model_step_30000'
        task_str = 'msfd_lv128,li'
        self.d_model_paths[task_str] = model_path

        # MSFD image translation sparse view 32 linear to multisegment BP 
        model_path = "pretrained_models/msfd_ss32,ms/serialized_model_step_90000"
        task_str = 'msfd_ss32,ms'
        self.d_model_paths[task_str] = model_path

        # MSFD image translation sparse view 64 linear to multisegment BP 
        model_path = "pretrained_models/msfd_ss64,ms/serialized_model_step_100000"
        task_str = 'msfd_ss64,ms'
        self.d_model_paths[task_str] = model_path

        # MSFD image translation sparse view 128 linear to multisegment BP 
        model_path = "pretrained_models/msfd_ss128,ms/serialized_model_step_95000"
        task_str = 'msfd_ss128,ms'
        self.d_model_paths[task_str] = model_path

        # SWFD image translation limited view 128 linear to multisegment BP
        model_path = "pretrained_models/swfd_lv128,li/serialized_model_step_15000"
        task_str = 'swfd_lv128,li'
        self.d_model_paths[task_str] = model_path

        # SWFD image translation limited view 128 semi-circle to semi-circle BP 
        model_path = "pretrained_models/swfd_lv128,sc/serialized_model_step_85000"
        task_str = 'swfd_lv128,sc'
        self.d_model_paths[task_str] = model_path

        # SWFD image translation sparse view 32 semi-circle to semi-circle BP 
        model_path = "pretrained_models/swfd_ss32,sc/serialized_model_step_55000"
        task_str = 'swfd_ss32,sc'
        self.d_model_paths[task_str] = model_path

        # SWFD image translation sparse view 64 semi-circle to semi-circle BP
        model_path = "pretrained_models/swfd_ss64,sc/serialized_model_step_110000"
        task_str = 'swfd_ss64,sc'
        self.d_model_paths[task_str] = model_path

        # SWFD image translation sparse view 128 semi-circle to semi-circle BP
        model_path = "pretrained_models/swfd_ss128,sc/serialized_model_step_125000"
        task_str = 'swfd_ss128,sc'
        self.d_model_paths[task_str] = model_path

        # SWFD image translation sparse view 32 multisegment to multisegment BP 
        model_path = "pretrained_models/swfd_ss32,ms/serialized_model_step_70000"
        task_str = 'swfd_ss32,ms'
        self.d_model_paths[task_str] = model_path

        # SWFD image translation sparse view 64 multisegment to multisegment BP
        model_path = "pretrained_models/swfd_ss64,ms/serialized_model_step_95000"
        task_str = 'swfd_ss64,ms'
        self.d_model_paths[task_str] = model_path

        # SWFD image translation sparse view 128 multisegment to multisegment BP
        model_path = "pretrained_models/swfd_ss128,ms/serialized_model_step_125000"
        task_str = 'swfd_ss128,ms'
        self.d_model_paths[task_str] = model_path

        # SCD image translation sparse view 64 ms to circle BP
        model_path = "pretrained_models/scd_ss128,ms/serialized_model_step_75000"
        task_str = 'scd_ss128,ms'
        self.d_model_paths[task_str] = model_path
        
        # SCD image translation sparse view 64 ms to circle BP
        model_path = "pretrained_models/scd_ss64,ms/serialized_model_step_80000"
        task_str = 'scd_ss64,ms'
        self.d_model_paths[task_str] = model_path

        # SCD image translation sparse view 32 ms to circle BP
        model_path = "pretrained_models/scd_ss32,ms/serialized_model_step_60000"
        task_str = 'scd_ss32,ms'
        self.d_model_paths[task_str] = model_path

        # SCD image translation sparse view 32 circle to circle BP
        model_path = "pretrained_models/scd_ss32,vc/serialized_model_step_65000"
        task_str = 'scd_ss32,vc'
        self.d_model_paths[task_str] = model_path

        # SCD image translation sparse view 64 circle to circle BP
        model_path = "pretrained_models/scd_ss64,vc/serialized_model_step_70000"
        task_str = 'scd_ss64,vc'
        self.d_model_paths[task_str] = model_path

        # SCD image translation sparse view 128 circle to circle BP
        model_path = "pretrained_models/scd_ss128,vc/serialized_model_step_65000"
        task_str = 'scd_ss128,vc'
        self.d_model_paths[task_str] = model_path

        # SCD image translation limited view 128 circle to circle BP
        model_path = "pretrained_models/scd_lv128,vc/serialized_model_step_75000"
        task_str = 'scd_lv128,vc'
        self.d_model_paths[task_str] = model_path

        # SCD image translation limited view 128 linear to circle BP
        model_path = "pretrained_models/scd_lv128,li/serialized_model_step_75000"
        task_str = 'scd_lv128,li'
        self.d_model_paths[task_str] = model_path

        # SCD image translation limited view 256 ms to circle BP
        model_path = "pretrained_models/scd_lv256,ms/serialized_model_step_80000"
        task_str = 'scd_lv256,ms'
        self.d_model_paths[task_str] = model_path

        #########################
        ## Image segmentation tasks

        # image segmentation sparse view 32 circle to circle BP labels 
        model_path = "pretrained_models/seg_ss32,vc/serialized_model_step_10100"
        task_str = 'seg_ss32,vc'
        self.d_model_paths[task_str] = model_path

        # image segmentation sparse view 64 circle to circle BP labels
        model_path = "pretrained_models/seg_ss64,vc/serialized_model_step_10000"
        task_str = 'seg_ss64,vc'
        self.d_model_paths[task_str] = model_path

        # image segmentation sparse view 128 circle to circle BP labels
        model_path = "pretrained_models/seg_ss128,vc/serialized_model_step_16400"
        task_str = 'seg_ss128,vc'
        self.d_model_paths[task_str] = model_path

        # image segmentation sparse view 128 multisegment to circle BP labels
        model_path = "pretrained_models/seg_ss128,ms/serialized_model_step_16600"
        task_str = 'seg_ss128,ms'
        self.d_model_paths[task_str] = model_path

        # image segmentation sparse view 64 multisegment to circle BP labels
        model_path = "pretrained_models/seg_ss64,ms/serialized_model_step_15900"
        task_str = 'seg_ss64,ms'
        self.d_model_paths[task_str] = model_path

        # image segmentation sparse view 32 multisegment to circle BP labels
        model_path = "pretrained_models/seg_ss32,ms/serialized_model_step_15300"
        task_str = 'seg_ss32,ms'
        self.d_model_paths[task_str] = model_path

        # image segmentation limited view 128 multisegment to circle BP labels
        model_path = "pretrained_models/seg_lv128,li/serialized_model_step_16500"
        task_str = 'seg_lv128,li'
        self.d_model_paths[task_str] = model_path

        # image segmentation limited view 128 circle to circle BP labels
        model_path = "pretrained_models/seg_lv128,vc/serialized_model_step_12400"
        task_str = 'seg_lv128,vc'
        self.d_model_paths[task_str] = model_path

        # image segmentation full view 1024 circle to circle BP labels
        model_path = "pretrained_models/seg_vc,vc/serialized_model_step_36900"
        task_str = 'seg_vc,vc'
        self.d_model_paths[task_str] = model_path

        model_path = "pretrained_models/seg_swfd_ms,ms/serialized_model_step_4700"
        task_str = 'seg_swfd_ms,ms'
        self.d_model_paths[task_str] = model_path

        model_path = "pretrained_models/seg_swfd_ss32,sc/serialized_model_step_5400"
        task_str = 'seg_swfd_ss32,sc'
        self.d_model_paths[task_str] = model_path

        model_path = "pretrained_models/seg_swfd_ss64,sc/serialized_model_step_4500"
        task_str = 'seg_swfd_ss64,sc'
        self.d_model_paths[task_str] = model_path

        model_path = "pretrained_models/seg_swfd_ss128,sc/serialized_model_step_3200"
        task_str = 'seg_swfd_ss128,sc'
        self.d_model_paths[task_str] = model_path

        model_path = "pretrained_models/seg_swfd_ss32,ms/serialized_model_step_2700"
        task_str = 'seg_swfd_ss32,ms'
        self.d_model_paths[task_str] = model_path

        model_path = "pretrained_models/seg_swfd_ss64,ms/serialized_model_step_2700"
        task_str = 'seg_swfd_ss64,ms'
        self.d_model_paths[task_str] = model_path

        model_path = "pretrained_models/seg_swfd_ss128,ms/serialized_model_step_2900"
        task_str = 'seg_swfd_ss128,ms'
        self.d_model_paths[task_str] = model_path

        model_path = "pretrained_models/seg_swfd_sc,sc/serialized_model_step_3500"
        task_str = 'seg_swfd_sc,sc'
        self.d_model_paths[task_str] = model_path

        model_path = "pretrained_models/seg_swfd_lv128,sc/serialized_model_step_4200"
        task_str = 'seg_swfd_lv128,sc'
        self.d_model_paths[task_str] = model_path

        model_path = "pretrained_models/seg_swfd_lv128,ms/serialized_model_step_2200"
        task_str = 'seg_swfd_lv128,ms'
        self.d_model_paths[task_str] = model_path

        model_path = "pretrained_models/seg_msfd_ss32,ms/serialized_model_step_3300"
        task_str = 'seg_msfd_ss32,ms'
        self.d_model_paths[task_str] = model_path

        model_path = "pretrained_models/seg_msfd_ss64,ms/serialized_model_step_3100"
        task_str = 'seg_msfd_ss64,ms'
        self.d_model_paths[task_str] = model_path

        model_path = "pretrained_models/seg_msfd_ss128,ms/serialized_model_step_3300"
        task_str = 'seg_msfd_ss128,ms'
        self.d_model_paths[task_str] = model_path

        model_path = "pretrained_models/seg_msfd_lv128,ms/serialized_model_step_4000"
        task_str = 'seg_msfd_lv128,ms'
        self.d_model_paths[task_str] = model_path
        

    def load_model(self, task_str):
        if str(task_str).lower() in self.list_task_str:
            os.system(f'git lfs pull -I {self.d_model_paths[task_str]}') ## pull model from git LFS 
        else:
            raise AssertionError(f'Not recognized task_str: {task_str}. Options are: {self.list_task_str}.')
        return tf.keras.models.load_model(self.d_model_paths[task_str], compile=False)

def save_single_sample_for_each_task_to_repo(datasets_parent_dir, sample_testdata_index_to_save=1000):
    '''This is the script used to save a sample for each task from the test set.'''
    if os.path.basename(os.getcwd()) == 'src':
        os.chdir('..')
    mpm_obj = Manage_Pretrained_Models()
    list_tasks = mpm_obj.list_task_str
    print(f'Task strings: {list_tasks}')
    d_fname_and_keys = {}
    for task_str in list_tasks:
        hpobj = load_model_hyperparameters(task_str=task_str, datasets_parent_dir=datasets_parent_dir)
        fname = hpobj.fname_h5
        ind = hpobj.inds_test[sample_testdata_index_to_save]
        if fname not in d_fname_and_keys:
            d_fname_and_keys[fname] = []

        for key in [hpobj.in_key, hpobj.out_key]:
            if isinstance(key, list):
                for i in range(len(key)):
                    pair = (key[i], ind)
                    if pair not in d_fname_and_keys[fname]:
                        d_fname_and_keys[fname].append(pair)
            else:
                pair = (key, ind)
                if pair not in d_fname_and_keys[fname]:
                    d_fname_and_keys[fname].append(pair)
    for fname in d_fname_and_keys.keys():
        print(f"There are {len(d_fname_and_keys[fname])} pairs for {fname}: {d_fname_and_keys[fname]}")

    for fname in d_fname_and_keys.keys():
        d = {}
        fname_out = os.path.basename(fname).replace('h5', 'p')
        path_out = os.path.join('data/sample_data', fname_out)
        with h5py.File(fname, 'r') as fh:
            for pair in d_fname_and_keys[fname]:
                key, ind = pair
                print(f"{key}, {ind}")
                d[key] = fh[key][ind,...]
        with open(path_out, 'wb') as fh:
            pickle.dump(d, fh)


