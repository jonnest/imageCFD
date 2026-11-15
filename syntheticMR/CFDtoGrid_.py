""" Student project - CFD to grid projection

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


# target = os.path.abspath('/Users/jonathan/Documents/applications/cfd/SampleUbend/sUbend_3/')    
# target = os.path.abspath('/Users/jonathan/Documents/applications/superres/pipe_1000/')    
# target = os.path.abspath('/Users/jonathan/Documents/applications/cfd/SampleUbend/ubend_1000_m1/')    
# target = os.path.abspath('/Users/jonathan/Documents/applications/cfd/SampleUbend/sUbend_re94/')    
target = os.path.abspath('/Users/jonathan/Documents/applications/cfd/SampleUbend/sUbend_re50/')    

cfd_folder = 'cfd'
# cfd_folder = 'cfd_vtk'

# for multiple time frames
nr_bins = 1

# isotropioc resolution
# image_resolution_list = np.array([0.5, 1, 2, 4])*1e-3

image_resolution_list = np.array([0.2, 0.4])*1e-3

for image_resolution in image_resolution_list:

    print("\nImage resolution:   ", image_resolution)
    # make data dir with resolution extension
    save_folder = f"data_r{int(image_resolution * 1e4):02d}" 
    os.makedirs(target+'/'+save_folder, exist_ok=True)

    # loop over time bins
    for time_index in range(nr_bins):

        ### get cfd files : adapt CFD folder names
        ### CHANGE for specific folder
        # flow_files = fetchFiles(target + f'/{cfd_folder}/*', ['averaged_'],['walls_'])
        # flow_files = fetchFiles(target + f'/{cfd_folder}/*', ['pipe_'],['walls_'])
        # flow_files = fetchFiles(target + f'/{cfd_folder}/*', ['ubend_'],['walls_'])
        # flow_files = fetchFiles(target + f'/{cfd_folder}/*', ['vessel_'],['walls_'])
        flow_files = fetchFiles(target + f'/{cfd_folder}/*', ['internal'],['walls_'])
        
        index = -1
        
        print("Read the following file  ", flow_files[index])
        U = np.empty(pv.read(flow_files[index])['U'].shape)
            
        # flow = pv.read(flow_files[time_index])
        flow = pv.read(flow_files[index])
        
        # scale flow up to resolution of image space in pixel with r=0.0004
        # after rescaling, CFD flow and MRI are in the same image space
        # flow = flow.scale(1/image_resolution)

        # Remove all cell data attributes from the mesh except for 'U' (velocity vector) and 'RST' (stress tensor)
        for attribute in flow.cell_data.keys():
            if attribute == 'U' or attribute == 'p':
                pass
            else:
                flow.cell_data.remove(attribute)
        flow['U']=U
        # add mask
        flow['mask']=np.ones((len(U)))
        
        # Padding around the image box in meters
        image_padding = 10e-3 #10e-3
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

        
        # Sample the mesh data onto the image grid
        image = image.sample(flow)

        # for slice plotting    
        y_slice = int(image.dimensions[-2]/2)
        z_slice = int(image.dimensions[-1]/2)
        
        show_y_slice = False
        show_z_slice = True
        # Reshape and extract relevant data arrays from the image object for velocity (U) and RST
        velocity_npy = np.reshape(image['U'],(image.dimensions+(image['U'].shape[-1],)), order='F')
        
        # mask_npy = np.reshape(image['vtkValidPointMask'],(image.dimensions),order='F')
        mask_npy = np.reshape(image['mask'],(image.dimensions),order='F')
        
        # pressure
        pressure_npy = np.reshape(image['p'],(image.dimensions),order='F')

        if show_y_slice == True:
            plt.imshow(velocity_npy[:,y_slice,:,2])
            plt.show()
            plt.imshow(pressure_npy[:,y_slice,:])
            plt.show()
            plt.imshow(mask_npy[:,y_slice,:])
            plt.show()
        elif show_z_slice == True:
            plt.imshow(velocity_npy[:,:,z_slice,0])   
            plt.show() 
            plt.imshow(pressure_npy[:,:,z_slice])
            plt.show()
            plt.imshow(mask_npy[:,:,z_slice])
            plt.show()
        
        # Save the array to a file
        np.save(f"{target}/{save_folder}/velocity.npy", velocity_npy)
        np.save(f"{target}/{save_folder}/mask.npy", mask_npy)
        np.save(f"{target}/{save_folder}/pressure.npy", pressure_npy)



    ########################################################################################
    # Load the array back
    # loaded_array = np.load("array.npy")
    # loaded_array = np.load(f"{target}/{save_folder}/pressure.npy")