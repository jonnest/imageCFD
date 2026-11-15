""" Generate synthetic 4d flow MRI data from CFD embedded background data

- Introduce various SNRs
- Different temporal resolutions
"""
import h5py
import numpy as np 
import matplotlib.pyplot as plt 
import pyvista as pv 
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage as ndi
import os,sys
import skimage
from scipy.ndimage import zoom as zoom
import sys,os,glob,re
import time

from gridOperators import make_odd, interpolate_3d_grid
from filters import *
from fileOperators import fetchFiles
from functionsRST import functionsRST


w_dir = os.path.abspath('/Users/jonathan/Documents/applications/cfd/eccomas_case/')    

r_iso = 0.4 # in mm
# temporal resolution of averaging
temporal_res_list = ['90']  # ['50', '70', '90']
# signal to noise ratio list
snr_list = [20]   # [10, 15, 20]
# spatial resolution list
spatial_list = [0.6]   # [0.4, 0.6, 0.8, 1, 1.2]



# downsampling factor from baseline resolution of r_iso
reduction_factor_list = r_iso / np.array(spatial_list)

t_start = time.time()

for temporal_res in temporal_res_list:
    for reduction_factor in reduction_factor_list:
        for snr in snr_list:
    
            print('\n----------------------------------------')
            print('Temporal resolution : ', temporal_res)
            print('Reduction factor :    ', reduction_factor)
            print('SNR :                 ', snr)
            print('----------------------------------------\n')
            
            vmax = 1  # Initialize the maximum velocity variable.
            extra_velocity_margin = 1  # Define an extra margin for the peak velocity to ensure coverage.

            # Calculate the encoding velocity (venc) with an extra margin and rounding to the nearest 0.5.
            venc = np.array([np.ceil(2 * vmax * extra_velocity_margin) / 2])/3

            # Set the signal-to-noise ratio (SNR) and the reduction factor for downsampling.
            new_voxel_size = "{0:.3g}".format(r_iso/reduction_factor)

            # Define the MRI encoding directions as unit vectors along the x, y, and z axes.
            enc_dirs = np.array([[1., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 1.]])

            # Calculate the encoding strength (kv_rec) based on the encoding velocity (venc).
            # It starts with 0 (no encoding) and then uses the formula π/venc for each direction.
            kv_rec = np.concatenate(([0], np.pi / venc))
            kv_rec = kv_rec[:, None]  # Add a new axis to make kv_rec a column vector.
            kv_rec = np.tile(kv_rec, (1, len(enc_dirs)))  # Repeat kv_rec for each encoding direction.
            
            # get files with embedded CFD for 3 encoding directions and all time bins with SNRinf 
            files_venc0 = fetchFiles(w_dir + f'/embedded_50ms/*', ['embdedded','hdf5'],['venc2','venc1'])
            files_venc1 = fetchFiles(w_dir + f'/embedded_50ms/*', ['embdedded','hdf5'],['venc2','venc0'])
            files_venc2 = fetchFiles(w_dir + f'/embedded_50ms/*', ['embdedded','hdf5'],['venc0','venc1'])

            mask = h5py.File(files_venc0[0])['3D_recon']['mask'][...]
            magnitude_ref = h5py.File(files_venc0[0])['3D_recon']['magnitude'][...]
            mean_magnitude = np.mean(magnitude_ref[mask>0.5])

            # Iterate over volume files matching a specific pattern, such as 'volume_'
            for volume_index,target in enumerate(files_venc0):

                image0 = h5py.File(files_venc0[volume_index])['3D_recon']['magnitude'][...] * np.exp(1j*(h5py.File(files_venc0[volume_index])['3D_recon']['phase'][...]))
                image1 = h5py.File(files_venc1[volume_index])['3D_recon']['magnitude'][...] * np.exp(1j*(h5py.File(files_venc1[volume_index])['3D_recon']['phase'][...]))
                image2 = h5py.File(files_venc2[volume_index])['3D_recon']['magnitude'][...] * np.exp(1j*(h5py.File(files_venc2[volume_index])['3D_recon']['phase'][...]))

                # Initialize an array to hold synthetic MRI data
                Skv = np.concatenate((image0[...,np.newaxis],
                                      image1[...,np.newaxis],
                                      image2[...,np.newaxis]),
                                      axis=-1)
                
                # Perform downsampling of MR data in fourier domain
                int_complex = []
                for direction in range(Skv.shape[-1]):
                    try:
                        int_complex = np.concatenate((int_complex,
                                                      interpolate_3d_grid(Skv[...,direction],reduction_factor, 
                                                                          method='fft',axes=(0,1,2),kwindow='hybrid',
                                                                          alpha=0.99)[...,np.newaxis]),
                                                     axis=-1)
                    except:
                        int_complex = interpolate_3d_grid(Skv[...,direction],reduction_factor, 
                                                          method='fft',axes=(0,1,2),kwindow='hybrid',
                                                          alpha=0.99)[...,np.newaxis]

                # Introduce noise to the embedded CFD data since flow field has SNRinf through divergence free CFD simulation
                # signal / SNR = noise
                int_complex +=  np.random.normal(0, mean_magnitude/snr, int_complex.shape) + \
                                1j*np.random.normal(0, mean_magnitude/snr, int_complex.shape)
                
                # Make directory
                save_dir = w_dir + f'/embedded_{new_voxel_size}mm_{temporal_res}ms_SNR{snr}'
                if not os.path.isdir(save_dir):    
                    # if the demo_folder2 directory is  
                    # not present then create it 
                    os.makedirs(save_dir) 
                
                # Save MR files
                file = h5py.File(save_dir + f'/embedded_{new_voxel_size}mm_{temporal_res}ms_SNR{snr}_t{volume_index}.hdf5','a')
                group = file.create_group('3D_recon')
                group['magnitude'] = np.abs(int_complex).astype('half')
                group['phase'] = np.angle(int_complex).astype('half')
                file.close()
                
                
t_end = time.time()
print('\nTotal time  ', (t_end-t_start)/60, ' min')