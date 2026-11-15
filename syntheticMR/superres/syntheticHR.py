'''  Time average the CFD simulations with kernel
- Crop velocity field to one vessel only !
- Select time average kernel and number of bins for 1 cardiac cycle (here = 1 second)
- Interpolate on high resolution grid
- Save as .h5 file
'''
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

from gridOperators import *
from filters import *
from fileOperators import *
from functionsRST import *


folder = 'geom3'
hr_folder = 'hr_data'
base_path = os.path.abspath(f'/Users/jonathan/Documents/applications/cfd/eccomas_case/CFD') 
out_path = os.path.abspath(f'/Users/jonathan/Documents/applications/superres/{folder}/time_data/') 
mri_path = os.path.abspath(f'/Users/jonathan/Documents/applications/cfd/eccomas_case/synthesizedMR') 

# time averaging in milli seconds, default: 50, 70, 90
time_avg = 50
nr_bins = 20 # default 20

mult_factor = 2
venc = 1
            
# isotropioc resolution
image_resolution = 0.4e-3 # 1.25e-3  # default 1.25e-3

# total of 100 simulation time steps, for selector of CFD time steps
time_scale = 100/nr_bins

# time averaging kernel
delta = {
    50 : [-2,-1,0,1,2],
    70 : [-3,-2,-1,0,1,2,3],
    90 : [-4,-3,-2,-1,0,1,2,3,4]
}

if not os.path.exists(out_path+'/'+hr_folder):
    os.makedirs(out_path+'/'+hr_folder)
    
# loop over time bins
for time_index in range(nr_bins):
    # simulation time 
    n_time = int(time_index * time_scale)
    delta_average = delta[time_avg]
    print("Time  ", n_time)
    
    # get cfd files : adapt CFD folder names
    flow_files = fetchFiles(base_path + f'/{folder}/*', ['vessel_'],['wall_'])
    U = np.empty(pv.read(flow_files[0])['U'].shape)
    
    # time averaging over the kernel
    for target_avg in delta_average:
        # print("averaging ", (n_time+target_avg)%len(flow_files))
        U+=pv.read(flow_files[(n_time+target_avg)%len(flow_files)])['U']/len(delta_average)

    mesh = pv.read(flow_files[0])
    # scale flow up to resolution of image space in pixel with r=0.0004
    # after rescaling, CFD flow and MRI are in the same image space 
    # mesh = mesh.scale(1/image_resolution)

    # Remove all cell data attributes from the mesh except for 'U' (velocity vector) and 'RST' (stress tensor)
    for attribute in mesh.cell_data.keys():
        if attribute == 'U' or attribute == 'RST':
            pass
        else:
            mesh.cell_data.remove(attribute)
    mesh['U']=U
    
    print('Prepare image')
    image_padding = 0e-3  # 10e-3 # Padding around the image box in meters
    # Calculate the image box dimensions and origin from the mesh bounds, adding padding
    image_box = np.array([mesh.bounds[1]-mesh.bounds[0]+image_padding,mesh.bounds[3]-mesh.bounds[2]+image_padding,mesh.bounds[5]-mesh.bounds[4]+image_padding])
    image_origin = np.array([mesh.bounds[0]-image_padding/2,mesh.bounds[2]-image_padding/2,mesh.bounds[4]-image_padding/2])

    # Ensure the image shape in each dimension is odd for FFT purposes
    print('\nImage shape (original)  ', np.array(image_box/image_resolution).astype('int'))
    image_shape = make_odd(np.array(image_box/image_resolution).astype('int'))
    print('\nImage shape (odd)       ', image_shape)

    # Create a new image data object and set its dimensions, origin, and spacing
    image = pv.ImageData()
    image.dimensions = image_shape
    image.origin = image_origin
    image.spacing = np.tile(image_resolution,3)

    # Sample the mesh data onto the image grid
    image = image.sample(mesh)

    # Reshape and extract relevant data arrays from the image object for velocity (U) and RST
    velocity_npy = np.reshape(image['U'],(image.dimensions+(image['U'].shape[-1],)), order='F')
    
    # Initialize the maximum velocity variable.
    vmax = np.abs(velocity_npy).max()
    vmean = (velocity_npy).mean()
    print("Mean velocity in field  ", vmean)
    print("Max absolute velocity in field  ", vmax)
    
    magnitude_npy = np.reshape(image['vtkValidPointMask'],(image.dimensions),order='F')


    # HR data 
    hr_dx = np.expand_dims(np.tile(image_resolution,3)*1000,axis=0)
    # hr_u = np.expand_dims(velocity_npy[...,0],axis=0)
    # hr_v = np.expand_dims(velocity_npy[...,1],axis=0)
    # hr_w = np.expand_dims(velocity_npy[...,2],axis=0)
    
    
    # Save high-resolution CFD data
    target_file = f'{out_path}/{hr_folder}/data_HR_r{int(image_resolution*10000)}_t{time_index}.h5'
    if os.path.exists(target_file):
        os.remove(target_file)
    file = h5py.File(target_file,'a')
    file.create_dataset('dx', data = hr_dx, dtype='half')
    file.create_dataset('u',  data = np.expand_dims(velocity_npy[...,0],axis=0), dtype='half')
    file.create_dataset('v',  data = np.expand_dims(velocity_npy[...,1],axis=0), dtype='half')
    file.create_dataset('w',  data = np.expand_dims(velocity_npy[...,2],axis=0), dtype='half')
    file.create_dataset('u_max', data = np.expand_dims(np.abs(velocity_npy[...,0]).max(),axis=0), dtype='half')
    file.create_dataset('v_max', data = np.expand_dims(np.abs(velocity_npy[...,1]).max(),axis=0), dtype='half')
    file.create_dataset('w_max', data = np.expand_dims(np.abs(velocity_npy[...,2]).max(),axis=0), dtype='half')
    file.create_dataset('mask',  data = np.expand_dims(magnitude_npy,axis=0), dtype='half')
    file.close()

    # concatenate into one large array
    if time_index == 0:
        hr_u = np.squeeze(velocity_npy[...,0])[np.newaxis,...]
        hr_v = np.squeeze(velocity_npy[...,1])[np.newaxis,...]
        hr_w = np.squeeze(velocity_npy[...,2])[np.newaxis,...]

    else:
        # For subsequent files, concatenate the new data volumes along the last axis.
        hr_u = np.concatenate((hr_u,np.squeeze(velocity_npy[...,0])[np.newaxis,...]),axis=0)
        hr_v = np.concatenate((hr_v,np.squeeze(velocity_npy[...,1])[np.newaxis,...]),axis=0)
        hr_w = np.concatenate((hr_w,np.squeeze(velocity_npy[...,2])[np.newaxis,...]),axis=0)


    # # save vtk file
    # fields = ['u', 'v', 'w']
    # grid = pv.ImageData()
    # grid.dimensions = [x_ for x_ in image.dimensions]
    # for i in range(3):
    #     grid[fields[i]] = np.reshape( velocity_npy[...,i].astype('double'),-1,order='F')
    # grid['mask'] = np.reshape( magnitude_npy.astype('double'),-1,order='F')
    # grid.save(os.path.join(out_path, f'data_HR_r{int(image_resolution*10000)}_t{time_index}.vtk'))

        
# Save time-averaged high-resolution CFD data - save all in one 
target_file = f'{out_path}/data_HR_r{int(image_resolution*10000)}.h5'
if os.path.exists(target_file):
    os.remove(target_file)
file = h5py.File(target_file,'a')
file.create_dataset('dx', data = hr_dx, dtype='half')
file.create_dataset('u',  data = hr_u, dtype='half')
file.create_dataset('v',  data = hr_v, dtype='half')
file.create_dataset('w',  data = hr_w, dtype='half')
file.create_dataset('u_max', data = np.expand_dims(np.abs(hr_u).max(),axis=0), dtype='half')
file.create_dataset('v_max', data = np.expand_dims(np.abs(hr_v).max(),axis=0), dtype='half')
file.create_dataset('w_max', data = np.expand_dims(np.abs(hr_w).max(),axis=0), dtype='half')
file.create_dataset('mask',  data = np.expand_dims(magnitude_npy,axis=0), dtype='half')
file.close()

    