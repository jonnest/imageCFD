""" Upsample 4d MRI flow field to high-resolution 0.4mm 
To match CFD (ground truth), afterwards use embedFlow.py
 
Save each velocity encoding direction separately
SNRinf as baseline reference of measurement

Running order
- synthesizedMR.py
- embedFlow.py 
- generateSyntheticMR.py

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
from fileOperators import *
from gridOperators import minmaxnorm, interpolate_3d_grid


# Load the data from a .h5 file
target = os.path.abspath('/Users/jonathan/Documents/applications/cfd/eccomas_case/')    
folder = 'synthesizedMR'

# Parameters for data processing
# This is important to artificially extend vasculature beyond the imaging window
# to make it easier to manually correct the boundaries of the vessel
pad_data = True #If true add padding to the edges of the array. Recommended: "True"
n_padding_voxels = 0 #Number of voxels to add along each of the spatial directions.
method = 'fft' #Method can be either 'fft' (zero-padding) or 'linear' (linear interpolation) 
up_factor = 2 #Upsampling factor

# get velocity field
flow_file = target+'/eccomas_case.h5'
loader = loadData()
image = loader.loadFullFlowField(flow_file)

#Generate padded image
if pad_data==True:
    image = np.pad(image,((n_padding_voxels,n_padding_voxels),
                          (n_padding_voxels,n_padding_voxels),
                          (n_padding_voxels,n_padding_voxels),
                          (0,0),(0,0)),mode='edge')

# reinterpolate along time dimension to get 50ms aquisition
# zoom factor (1,1.05), first dimension unchanged, time dimension interpolated by factor 1.05
# increasing the number of time bins to 20
# order 1 = linear interpolation
image = np.reshape(zoom(np.reshape(image,(-1,19),order='F')[:,:],(1,1.05),order=1),
                   (image.shape[:-1])+(20,),order='F')

# make directory 
save_dir = target + '/' + folder
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# loop over velocity encoding directions x,y,z
for i in range(image.shape[-2]):
    # loop over time bins
    for j in range(image.shape[-1]):
        # upsample
        tmp_ = interpolate_3d_grid(image[...,i,j], up_factor, method=method)
        
        file_path = save_dir + f'/0.4mm_50ms_SNRinf_venc{i}_t{j}.h5'
        if os.path.exists(file_path):
            os.remove(file_path)
        file = h5py.File(file_path,'a')
        group = file.create_group('3D_recon')
        
        # image is complex : absolute=magnitude, angle=phase
        group['magnitude'] = minmaxnorm(np.abs(tmp_)).astype('half')
        group['phase'] = np.angle(tmp_).astype('half')
        file.close()

        grid = pv.ImageData()
        grid.dimensions = [x_ for x_ in tmp_.shape]
        grid['magnitude'] = np.reshape(minmaxnorm(np.abs(tmp_)),-1,order='F')
        grid['phase'] = np.reshape(np.angle(tmp_),-1,order='F')
        grid.save(save_dir + f'/0.4mm_50ms_SNRinf_venc{i}_t{j}.vtk')