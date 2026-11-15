""" Embedding the CFD simulation into the MRI background

- average velocity 
- crop down around one geometry
- embedd CFD into background of MRI

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
from fileOperators import fetchFiles
from gridOperators import *


target = os.path.abspath('/Users/jonathan/Documents/applications/cfd/eccomas_case/')    
mr_folder = 'synthesizedMR'
cfd_folder = 'CFD'

geom_index = 1 
# time averaging in milli seconds, default: 50, 70, 90
time_avg = 50
nr_bins = 20 # default 20

mult_factor = 2
venc = 1.
            
# isotropioc resolution
image_resolution = 0.4 * 1e-3
# total of 100 simulation time steps, for selector of CFD time steps
time_scale = 100/nr_bins

# time averaging kernel
delta = {
    50 : [-2,-1,0,1,2],
    70 : [-3,-2,-1,0,1,2,3],
    90 : [-4,-3,-2,-1,0,1,2,3,4]
}

# Define the MRI encoding directions as unit vectors along the x, y, and z axes.
enc_dirs = np.array([[1., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]])

# Calculate the encoding strength (kv_rec) based on the encoding velocity (venc).
# It starts with 0 (no encoding) and then uses the formula π/venc for each direction.
kv_rec = np.concatenate(([0], np.pi / np.array([venc])))
kv_rec = kv_rec[:, None]  # Add a new axis to make kv_rec a column vector.
kv_rec = np.tile(kv_rec, (1, len(enc_dirs)))  # Repeat kv_rec for each encoding direction.

# make directory 
save_dir = target + f'/embedded_{time_avg}ms'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# loop over time bins
for time_index in range(nr_bins):
    # initalize list to save all 3 components
    magnitude_list = []
    phase_list = []
        
    # loop over velocity encodings        
    for venc_index in range(3):
        print("venc index ", venc_index)
        mri = h5py.File(target + f'/{mr_folder}/0.4mm_50ms_SNRinf_venc{venc_index}_t{time_index}.hdf5')
        
        magnitude_full = mri['3D_recon']['magnitude'][...]
        phase_full = mri['3D_recon']['phase'][...]
        mask_full = np.empty(magnitude_full.shape)

        # simulation time 
        n_time = int(time_index * time_scale)
        delta_average = delta[time_avg]
        # get cfd files : adapt CFD folder names
        flow_files = fetchFiles(target + f'/{cfd_folder}/geom{geom_index+1}/*', ['vessel_'],['wall_'])
        
        U = np.empty(pv.read(flow_files[0])['U'].shape)
        
        # time averaging over the kernel
        for target_avg in delta_average:
            U+=pv.read(flow_files[(n_time+target_avg)%len(flow_files)])['U']/len(delta_average)
                

        flow = pv.read(flow_files[0])
        # scale flow up to resolution of image space in pixel with r=0.0004
        # after rescaling, CFD flow and MRI are in the same image space
        flow = flow.scale(1/image_resolution)

        # Remove all cell data attributes from the mesh except for 'U' (velocity vector) and 'RST' (stress tensor)
        for attribute in flow.cell_data.keys():
            if attribute == 'U' or attribute == 'RST':
                pass
            else:
                flow.cell_data.remove(attribute)
        flow['U']=U
        
        image_padding = 0  # 10e-3 # Padding around the image box in meters
        # Calculate the image box dimensions and origin from the mesh bounds, adding padding
        image_box = np.array([flow.bounds[1]-flow.bounds[0]+image_padding,flow.bounds[3]-flow.bounds[2]+image_padding,flow.bounds[5]-flow.bounds[4]+image_padding])
        image_origin = np.array([flow.bounds[0]-image_padding/2,flow.bounds[2]-image_padding/2,flow.bounds[4]-image_padding/2])
        print("image resolution ", image_resolution)
        # Ensure the image shape in each dimension is odd for FFT purposes
        image_shape = make_odd(np.array(image_box/image_resolution).astype('int'))

        # Create a new image data object and set its dimensions, origin, and spacing
        image = pv.ImageData()
        image.dimensions = image_shape
        image.origin = image_origin
        image.spacing = np.tile(image_resolution,3)

        1/0
        # Sample the mesh data onto the image grid
        image = image.sample(flow)