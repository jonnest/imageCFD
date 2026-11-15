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


files = [
            #'0.4mm_50ms_SNRinf_venc0_t0.hdf5',
            'embedded_50ms/embedded_0.4mm_50ms_SNRinf_venc0_t0.hdf5'
    
        ]
       
                
                
r_iso = 0.4 * 1e-3

for file in files:
    print(f'\n{file}')
    with h5py.File(file, 'r') as f: 
        try:
            data = f['3D_recon']['real'][:] + 1j*f['3D_recon']['imag'][:]
            data = np.squeeze(data)
            f.close()
            # vel_mag = np.sqrt(np.sum(np.square(np.angle(np.squeeze(data))), axis=3))
            vel_x = np.angle(np.squeeze(data))
            # print(f"Loaded {mr_file} data in real-imaginary format")
        except:
            data = f['3D_recon']['phase'][:]
            data = np.squeeze(data)
            f.close()
            
            # vel_mag = np.sqrt(np.sum(np.square(np.squeeze(data)), axis=3))
            vel_x = np.squeeze(data)
            
            # print(f"Loaded {mr_files} data in magnitude-phase format")
    
    # save data
    flow_im = pv.ImageData()
    flow_im.dimensions = (vel_x.shape[0]+1,
                          vel_x.shape[1]+1,
                          vel_x.shape[2]+1)
    
    print('flow image dims ', flow_im.dimensions)
    flow_im.cell_data['vel'] = np.reshape(vel_x, -1, order='F').astype('float')
    flow_im.spacing = (r_iso,r_iso,r_iso)
    name_tag = file.split('.hdf5')[0]
    flow_im.save(name_tag + '.vtk')
    