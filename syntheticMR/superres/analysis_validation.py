''' Validation of 4DFlowNet

- compare HR, inference
- compare LR with same upsampled matrix size !
- linearly interpolated
- compare to b-spline interpolation

SR Input
(1, nx, ny, nz)

TODO : explain how data has to be saved to use script

'''

import h5py
import pyvista as pv
import numpy as np 
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt 
import sys,os,glob,re
from scipy.ndimage import zoom as zoom
from skimage.transform import resize
from IPython.display import Image as ipImage
from scipy.ndimage import binary_erosion

from gridOperators import *
from filters import *
from fileOperators import *
from functionsRST import *
import h5_utils as h5utils
from flow_field_metrics import *
from plot_functions import *

folder = 'geom3'
base_path = f'/Users/jonathan/Documents/applications/superres/{folder}/time_data/'

# targetSNR 
snr_list = [20]

interpolated_data_exists = False # True
read_only_two = True
num_frames = 2 # 2, 20

image_resolution = 0.4e-3
reduction_factor = 0.5

hr_folder = 'hr_data'
lr_string = '_LR_'          # 'data_LR_'
hr_string = '_HR_'          # 'data_HR_'
up_string = '_upsampled_'   # '_upsampled_'
lin_string = '_linear_'      
cub_string = '_cubic_'   
fft_string = '_fft_'   

# z coordinate for slice computation
# oliver test cases: 12
# geom1: ?
# geom2: ?
# geom3: ?
# geom4: ?

z_slice = 28

### order in plots 
# superres upsampled
# lin
# cub
# fft - zerofilld
# ground truth
name_list = ['super-resolution', 'linear', 'cubic', 'zerofilled']
save_string = ['sr', 'lin', 'cub', 'fft']
directions = ['x', 'y', 'z']

# loop over available SNR evaluations
for snr in snr_list:
    print(f'\n*** Start analysis for SNR {snr} ! ***\n')
    
    hr_noise_folder = f'hr_data_snr{snr}'
    lr_folder = f'lr_data_snr{snr}'
    sr_folder = f'sr_data_snr{snr}'
    lin_folder = f'lin_data_snr{snr}'
    cub_folder = f'cub_data_snr{snr}'
    fft_folder = f'fft_data_snr{snr}'

    # create folders for the plots, snr and the subdivided into snr-prediction specific
    # Create the main directory for each SNR
    snr_dir = os.path.join(base_path, f'plots_snr_{snr}')
    os.makedirs(snr_dir, exist_ok=True)

    # Create subdirectories for sr, lin, cub, and fft inside the main snr folder
    sr_dir = os.path.join(snr_dir, f'plots_sr_snr{snr}')
    lin_dir = os.path.join(snr_dir, f'plots_lin_snr{snr}')
    cub_dir = os.path.join(snr_dir, f'plots_cub_snr{snr}')
    fft_dir = os.path.join(snr_dir, f'plots_fft_snr{snr}')

    snr_plot_dirs = [sr_dir, lin_dir, cub_dir, fft_dir]
    
    os.makedirs(sr_dir, exist_ok=True)
    os.makedirs(lin_dir, exist_ok=True)
    os.makedirs(cub_dir, exist_ok=True)
    os.makedirs(fft_dir, exist_ok=True)
    
    # ------------------------------------------------------------------
    # read data over time
    # ------------------------------------------------------------------
    for volume_index,file in enumerate(fetchFiles(os.path.join(base_path,lr_folder) + '/*',[lr_string], ['.vtk'])):
        
        # print("\nvolume_index  ", volume_index)
        # Open the current h5 file in read mode.
        data = h5py.File(file,'r')
        # print("LR file ", target.split('/')[-1] )
        case_name = os.path.splitext(os.path.basename(file))[0]
        # If this is the first file, initialize the data arrays by loading and squeezing
        # single data volumes from the file. `np.newaxis` is used to add a new axis,
        # preparing for concatenation along this axis for subsequent files.
        if volume_index == 0:
            ul = np.squeeze(data['u'][...])[...,np.newaxis]
            vl = np.squeeze(data['v'][...])[...,np.newaxis]
            wl = np.squeeze(data['w'][...])[...,np.newaxis]
            mu = np.squeeze(data['mag_u'][...])[...,np.newaxis]
            mv = np.squeeze(data['mag_v'][...])[...,np.newaxis]
            mw = np.squeeze(data['mag_w'][...])[...,np.newaxis]

            if interpolated_data_exists==False:
                # linearly interpolated data
                u_lin = interpolate_3d_grid(ul[...,volume_index], mult_factor=2, method='linear',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis]
                v_lin = interpolate_3d_grid(vl[...,volume_index], mult_factor=2, method='linear',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis]
                w_lin = interpolate_3d_grid(wl[...,volume_index], mult_factor=2, method='linear',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis]

                # zero filling
                u_fft = np.real(interpolate_3d_grid(ul[...,volume_index], mult_factor=2, method='fft',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis])
                v_fft = np.real(interpolate_3d_grid(vl[...,volume_index], mult_factor=2, method='fft',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis])
                w_fft = np.real(interpolate_3d_grid(wl[...,volume_index], mult_factor=2, method='fft',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis])

                # cubic interpolation
                u_cub = interpolate_3d_grid(ul[...,volume_index], mult_factor=2, method='cubic',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis]
                v_cub = interpolate_3d_grid(vl[...,volume_index], mult_factor=2, method='cubic',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis]
                w_cub = interpolate_3d_grid(wl[...,volume_index], mult_factor=2, method='cubic',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis]

        else:
            # For subsequent files, concatenate the new data volumes along the last axis.
            ul = np.concatenate((ul,np.squeeze(data['u'][...])[...,np.newaxis]),axis=-1)
            vl = np.concatenate((vl,np.squeeze(data['v'][...])[...,np.newaxis]),axis=-1)
            wl = np.concatenate((wl,np.squeeze(data['w'][...])[...,np.newaxis]),axis=-1)
            mu = np.concatenate((mu,np.squeeze(data['mag_u'][...])[...,np.newaxis]),axis=-1)
            mv = np.concatenate((mv,np.squeeze(data['mag_v'][...])[...,np.newaxis]),axis=-1)
            mw = np.concatenate((mw,np.squeeze(data['mag_w'][...])[...,np.newaxis]),axis=-1)  

            if interpolated_data_exists==False:
                
                # linearly interpolated data
                u_lin = np.concatenate((u_lin,interpolate_3d_grid(ul[...,volume_index], mult_factor=2, method='linear',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis]),axis=-1) 
                v_lin = np.concatenate((v_lin,interpolate_3d_grid(vl[...,volume_index], mult_factor=2, method='linear',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis]),axis=-1) 
                w_lin = np.concatenate((w_lin,interpolate_3d_grid(wl[...,volume_index], mult_factor=2, method='linear',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis]),axis=-1) 

                # zero filling
                u_fft = np.concatenate((u_fft,np.real(interpolate_3d_grid(ul[...,volume_index], mult_factor=2, method='fft',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis])),axis=-1) 
                v_fft = np.concatenate((v_fft,np.real(interpolate_3d_grid(vl[...,volume_index], mult_factor=2, method='fft',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis])),axis=-1) 
                w_fft = np.concatenate((w_fft,np.real(interpolate_3d_grid(wl[...,volume_index], mult_factor=2, method='fft',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis])),axis=-1) 

                # cubic interpolation
                u_cub = np.concatenate((u_cub,interpolate_3d_grid(ul[...,volume_index], mult_factor=2, method='cubic',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis]),axis=-1) 
                v_cub = np.concatenate((v_cub,interpolate_3d_grid(vl[...,volume_index], mult_factor=2, method='cubic',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis]),axis=-1) 
                w_cub = np.concatenate((w_cub,interpolate_3d_grid(wl[...,volume_index], mult_factor=2, method='cubic',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis]),axis=-1) 

            if read_only_two:
                if volume_index==1:
                    break
                
        # Store velocity encoding values in a list for further use.
        venc = [data['venc_u'][...],data['venc_v'][...],data['venc_w'][...]]
        # print("venc ", venc)
        # Close the current h5 file to free resources.
        data.close()   
    print("Read LR files, nr                ", volume_index)
        
    for volume_index,file in enumerate(fetchFiles(os.path.join(base_path,sr_folder) + '/*',[up_string], ['.vtk'])):
        data = h5py.File(file,'r')
        if volume_index == 0:
            # Initialize high-resolution data arrays for the first file.
            us = np.squeeze(data['u'][...])[...,np.newaxis]
            vs = np.squeeze(data['v'][...])[...,np.newaxis]
            ws = np.squeeze(data['w'][...])[...,np.newaxis]
        else:
            # Concatenate new data volumes to the existing arrays for subsequent files.
            us = np.concatenate((us,np.squeeze(data['u'][...])[...,np.newaxis]),axis=-1)
            vs = np.concatenate((vs,np.squeeze(data['v'][...])[...,np.newaxis]),axis=-1)
            ws = np.concatenate((ws,np.squeeze(data['w'][...])[...,np.newaxis]),axis=-1)  
            
            if read_only_two:
                if volume_index==1:
                    break
        # Close the current h5 file to free resources.
        data.close()    
    print("Read SR files, nr                ", volume_index)

    for volume_index,file in enumerate(fetchFiles(os.path.join(base_path,hr_folder) + '/*',[hr_string], ['.vtk'])):
        data = h5py.File(file,'r')
        # NOTE : error can occur if looping over file which contains all frames
        if volume_index == 0:
            # Initialize high-resolution data arrays for the first file.
            ugt = np.squeeze(data['u'][...])[...,np.newaxis]
            vgt = np.squeeze(data['v'][...])[...,np.newaxis]
            wgt = np.squeeze(data['w'][...])[...,np.newaxis]
            mask = np.squeeze(data['mask'][...])[...,np.newaxis]
        else:
            # Concatenate new data volumes to the existing arrays for subsequent files.
            ugt = np.concatenate((ugt,np.squeeze(data['u'][...])[...,np.newaxis]),axis=-1)
            vgt = np.concatenate((vgt,np.squeeze(data['v'][...])[...,np.newaxis]),axis=-1)
            wgt = np.concatenate((wgt,np.squeeze(data['w'][...])[...,np.newaxis]),axis=-1)
            mask = np.concatenate((mask,np.squeeze(data['mask'][...])[...,np.newaxis]),axis=-1)
            
            if read_only_two:
                if volume_index==1:
                    break
        # Close the current h5 file to free resources.
        data.close()   
    print("Read HR files, nr                ", volume_index)

    for volume_index,file in enumerate(fetchFiles(os.path.join(base_path,hr_noise_folder) + '/*',[hr_string], ['.vtk'])):
        data = h5py.File(file,'r')
        # NOTE : error can occur if looping over file which contains all frames
        if volume_index == 0:
            # Initialize high-resolution data arrays for the first file.
            ugtn = np.squeeze(data['u'][...])[...,np.newaxis]
            vgtn = np.squeeze(data['v'][...])[...,np.newaxis]
            wgtn = np.squeeze(data['w'][...])[...,np.newaxis]
            # mask = np.squeeze(data['mask'][...])[...,np.newaxis]
        else:
            # Concatenate new data volumes to the existing arrays for subsequent files.
            ugtn = np.concatenate((ugtn,np.squeeze(data['u'][...])[...,np.newaxis]),axis=-1)
            vgtn = np.concatenate((vgtn,np.squeeze(data['v'][...])[...,np.newaxis]),axis=-1)
            wgtn = np.concatenate((wgtn,np.squeeze(data['w'][...])[...,np.newaxis]),axis=-1)
            
            if read_only_two:
                if volume_index==1:
                    break
            # mask = np.concatenate((mask,np.squeeze(data['mask'][...])[...,np.newaxis]),axis=-1)
        # Close the current h5 file to free resources.
        data.close()   
    print("Read HR noise files, nr          ", volume_index)
    
    # if the interpolated data is already saved as files - read from .h5 files - faster than computing
    if interpolated_data_exists==True:
        for volume_index,file in enumerate(fetchFiles(os.path.join(base_path,lin_folder) + '/*',[lin_string], ['.vtk'])):
            data = h5py.File(file,'r')
            if volume_index == 0:
                # Initialize high-resolution data arrays for the first file.
                u_lin = np.squeeze(data['u'][...])[...,np.newaxis]
                v_lin = np.squeeze(data['v'][...])[...,np.newaxis]
                w_lin = np.squeeze(data['w'][...])[...,np.newaxis]
            else:
                # Concatenate new data volumes to the existing arrays for subsequent files.
                u_lin = np.concatenate((u_lin,np.squeeze(data['u'][...])[...,np.newaxis]),axis=-1)
                v_lin = np.concatenate((v_lin,np.squeeze(data['v'][...])[...,np.newaxis]),axis=-1)
                w_lin = np.concatenate((w_lin,np.squeeze(data['w'][...])[...,np.newaxis]),axis=-1)
                
            if read_only_two:
                if volume_index==1:
                    break
            # Close the current h5 file to free resources.
            data.close()   
        print("Read Linear files, nr            ", volume_index)

        for volume_index,file in enumerate(fetchFiles(os.path.join(base_path,cub_folder) + '/*',[cub_string], ['.vtk'])):
            data = h5py.File(file,'r')
            if volume_index == 0:
                # Initialize high-resolution data arrays for the first file.
                u_cub = np.squeeze(data['u'][...])[...,np.newaxis]
                v_cub = np.squeeze(data['v'][...])[...,np.newaxis]
                w_cub = np.squeeze(data['w'][...])[...,np.newaxis]
            else:
                # Concatenate new data volumes to the existing arrays for subsequent files.
                u_cub = np.concatenate((u_cub,np.squeeze(data['u'][...])[...,np.newaxis]),axis=-1)
                v_cub = np.concatenate((v_cub,np.squeeze(data['v'][...])[...,np.newaxis]),axis=-1)
                w_cub = np.concatenate((w_cub,np.squeeze(data['w'][...])[...,np.newaxis]),axis=-1)

            if read_only_two:
                if volume_index==1:
                    break
            # Close the current h5 file to free resources.
            data.close()   
        print("Read Cubic files, nr             ", volume_index)

        for volume_index,file in enumerate(fetchFiles(os.path.join(base_path,fft_folder) + '/*',[fft_string], ['.vtk'])):
            data = h5py.File(file,'r')
            if volume_index == 0:
                # Initialize high-resolution data arrays for the first file.
                u_fft = np.squeeze(data['u'][...])[...,np.newaxis]
                v_fft = np.squeeze(data['v'][...])[...,np.newaxis]
                w_fft = np.squeeze(data['w'][...])[...,np.newaxis]
            else:
                # Concatenate new data volumes to the existing arrays for subsequent files.
                u_fft = np.concatenate((u_fft,np.squeeze(data['u'][...])[...,np.newaxis]),axis=-1)
                v_fft = np.concatenate((v_fft,np.squeeze(data['v'][...])[...,np.newaxis]),axis=-1)
                w_fft = np.concatenate((w_fft,np.squeeze(data['w'][...])[...,np.newaxis]),axis=-1)
                
            if read_only_two:
                if volume_index==1:
                    break
            # Close the current h5 file to free resources.
            data.close()   
        print("Read FFT (Zerofilled) files, nr  ", volume_index)


    # print("\nCheck shapes of data")
    # print("shape us     ", us.shape)
    # print("shape ugt    ", ugt.shape)
    # print("shape ul     ", ul.shape)
    # print("shape u_lin  ", u_lin.shape)
    # print("shape u_cub  ", u_cub.shape)
    # print("shape u_fft  ", u_fft.shape)

    # ------------------------------------------------------------------
    # save interpolations (linear, cubic) and zerofill as .h5 files
    # ------------------------------------------------------------------
    if interpolated_data_exists==False:
        
        new_name = case_name.replace(lr_string, '_linear_')
        os.makedirs(os.path.join(base_path, lin_folder), exist_ok=True)
        for i in range(num_frames):
            output_filename = os.path.join(base_path, lin_folder, f'data_linear_t{i}.h5')
            if os.path.exists(output_filename):
                os.remove(output_filename)
            # Save the downsampled image
            h5utils.save_to_h5(output_filename, "u", u_lin[...,i])
            h5utils.save_to_h5(output_filename, "v", v_lin[...,i])
            h5utils.save_to_h5(output_filename, "w", w_lin[...,i])
            if read_only_two:
                if volume_index==1:
                    break
                
        new_name = case_name.replace(lr_string, '_cubic_')
        os.makedirs(os.path.join(base_path, cub_folder), exist_ok=True)
        for i in range(num_frames):
            output_filename = os.path.join(base_path, cub_folder, f'data_cubic_t{i}.h5')
            if os.path.exists(output_filename):
                os.remove(output_filename)
            # Save the downsampled image
            h5utils.save_to_h5(output_filename, "u", u_cub[...,i])
            h5utils.save_to_h5(output_filename, "v", v_cub[...,i])
            h5utils.save_to_h5(output_filename, "w", w_cub[...,i])
            if read_only_two:
                if volume_index==1:
                    break
                
        new_name = case_name.replace(lr_string, '_fft_')
        os.makedirs(os.path.join(base_path, fft_folder), exist_ok=True)
        for i in range(num_frames):
            output_filename = os.path.join(base_path, fft_folder, f'data_fft_t{i}.h5')
            if os.path.exists(output_filename):
                os.remove(output_filename)
            # Save the downsampled image
            h5utils.save_to_h5(output_filename, "u", u_fft[...,i])
            h5utils.save_to_h5(output_filename, "v", v_fft[...,i])
            h5utils.save_to_h5(output_filename, "w", w_fft[...,i])
            if read_only_two:
                if volume_index==1:
                    break
    # ------------------------------------------------------------------
    # data prep for resizing to same matrix sizes
    # ------------------------------------------------------------------

    # add a new axis to the data for stacking velocity components
    us = us[np.newaxis,...]
    vs = vs[np.newaxis,...]
    ws = ws[np.newaxis,...]

    ugt = ugt[np.newaxis,...]
    vgt = vgt[np.newaxis,...]
    wgt = wgt[np.newaxis,...]

    ugtn = ugtn[np.newaxis,...]
    vgtn = vgtn[np.newaxis,...]
    wgtn = wgtn[np.newaxis,...]
    
    u_lin = u_lin[np.newaxis,...]
    v_lin = v_lin[np.newaxis,...]
    w_lin = w_lin[np.newaxis,...]

    u_cub = u_cub[np.newaxis,...]
    v_cub = v_cub[np.newaxis,...]
    w_cub = w_cub[np.newaxis,...]

    u_fft = u_fft[np.newaxis,...]
    v_fft = v_fft[np.newaxis,...]
    w_fft = w_fft[np.newaxis,...]

    # concatenate x,y,z in second dimension
    target = np.concatenate((us,vs,ws),axis=0)
    target_gt = np.concatenate((ugt,vgt,wgt),axis=0)
    target_gtn = np.concatenate((ugtn,vgtn,wgtn),axis=0)
    lin = np.concatenate((u_lin,v_lin,w_lin),axis=0)
    cub = np.concatenate((u_cub,v_cub,w_cub),axis=0)
    fft = np.concatenate((u_fft,v_fft,w_fft),axis=0)


    # print("\nBefore resize target shape:   ", target.shape)
    # print("Target ground truth shape:    ", target_gt.shape)

    # resize like this but for every frame in the dataset separately
    resized_target = np.zeros(target_gt.shape)
    resized_lin = np.zeros(target_gt.shape)
    resized_cub = np.zeros(target_gt.shape)
    resized_fft = np.zeros(target_gt.shape)

    for i in range(target.shape[-1]):
        resized_target[...,i] = resize(target[...,i], target_gt[...,i].shape, order=3, mode='constant', anti_aliasing=True)
        resized_lin[...,i] = resize(lin[...,i], target_gt[...,i].shape, order=3, mode='constant', anti_aliasing=True)
        resized_cub[...,i] = resize(cub[...,i], target_gt[...,i].shape, order=3, mode='constant', anti_aliasing=True)
        resized_fft[...,i] = resize(fft[...,i], target_gt[...,i].shape, order=3, mode='constant', anti_aliasing=True)


    # print("\nResized target shape:         ", resized_target.shape)
    # print( "Target ground truth shape:    ", target_gt.shape)
    # print( "Resized target LR shape:      ", resized_lin.shape)

    # ------------------------------------------------------------------
    # get binary mask - mask domain
    # ------------------------------------------------------------------
    # get binary mask
    binary_mask = np.zeros(mask.shape,dtype=bool)
    binary_mask[mask>0.5] = True
    binary_mask_non_fluid = np.logical_not(binary_mask)
    print("\nBinary mask shape:            ", binary_mask.shape)

    # compute velocity magnitude
    norm_gt = np.sqrt(np.sum(target_gt**2,axis=0))
    norm_gtn = np.sqrt(np.sum(target_gtn**2,axis=0))
    norm_target = np.sqrt(np.sum(resized_target**2,axis=0))
    norm_lin = np.sqrt(np.sum(resized_lin**2,axis=0))
    norm_cub = np.sqrt(np.sum(resized_cub**2,axis=0))
    norm_fft = np.sqrt(np.sum(resized_fft**2,axis=0))

    # mask non-velocity field
    nf_masked_target = resized_target[:,binary_mask_non_fluid]
    nf_masked_target_gt = target_gt[:,binary_mask_non_fluid]
    nf_masked_target_gtn = target_gtn[:,binary_mask_non_fluid]
    nf_masked_lin = resized_lin[:,binary_mask_non_fluid]
    nf_masked_cub = resized_cub[:,binary_mask_non_fluid]
    nf_masked_fft = resized_fft[:,binary_mask_non_fluid]

    # mask complete velocity field - keep all velocity components
    masked_field = resized_target[:,binary_mask]
    masked_field_gt = target_gt[:,binary_mask]
    masked_field_gtn = target_gtn[:,binary_mask]
    masked_field_lin = resized_lin[:,binary_mask]
    masked_field_cub = resized_cub[:,binary_mask]
    masked_field_fft = resized_fft[:,binary_mask]
    
    # apply binary max to each frame of the array resized_target
    # the frame number is the last dimension (x,y,z, frame_t)
    masked_velocity = []
    masked_velocity_gt = []
    masked_velocity_gtn = []
    masked_velocity_lin = []
    masked_velocity_cub = []
    masked_velocity_fft = []

    nf_masked_velocity = []
    nf_masked_velocity_gt = []
    nf_masked_velocity_gtn = []
    nf_masked_velocity_lin = []
    nf_masked_velocity_cub = []
    nf_masked_velocity_fft = []
    
    # save masked velocity magnitude - loop over frames
    for i in range(binary_mask.shape[-1]):
        masked_velocity.append(norm_target[binary_mask[...,i],i])
        masked_velocity_gt.append(norm_gt[binary_mask[...,i],i])
        masked_velocity_gtn.append(norm_gtn[binary_mask[...,i],i])
        masked_velocity_lin.append(norm_lin[binary_mask[...,i],i])
        masked_velocity_cub.append(norm_cub[binary_mask[...,i],i])
        masked_velocity_fft.append(norm_fft[binary_mask[...,i],i])

        nf_masked_velocity.append(norm_target[binary_mask_non_fluid[...,i],i])
        nf_masked_velocity_gt.append(norm_gt[binary_mask_non_fluid[...,i],i])
        nf_masked_velocity_gtn.append(norm_gtn[binary_mask_non_fluid[...,i],i])
        nf_masked_velocity_lin.append(norm_lin[binary_mask_non_fluid[...,i],i])
        nf_masked_velocity_cub.append(norm_cub[binary_mask_non_fluid[...,i],i])
        nf_masked_velocity_fft.append(norm_fft[binary_mask_non_fluid[...,i],i])
        
    # Initialize an output array with the same shape as norm_target
    shape_masked_velocity = np.zeros_like(norm_target)
    shape_masked_velocity_gt = np.zeros_like(norm_target)
    shape_masked_velocity_gtn = np.zeros_like(norm_target)
    shape_masked_velocity_lin = np.zeros_like(norm_target)
    shape_masked_velocity_cub = np.zeros_like(norm_target)
    shape_masked_velocity_fft = np.zeros_like(norm_target)

    # Loop over frames
    for i in range(binary_mask.shape[-1]):
        # Apply the mask: wherever the binary mask is True, retain the value from norm_target
        shape_masked_velocity[..., i] = np.where(binary_mask[..., i], norm_target[..., i], 0)
        shape_masked_velocity_gt[..., i] = np.where(binary_mask[..., i], norm_gt[..., i], 0)
        shape_masked_velocity_gtn[..., i] = np.where(binary_mask[..., i], norm_gtn[..., i], 0)
        shape_masked_velocity_lin[..., i] = np.where(binary_mask[..., i], norm_lin[..., i], 0)
        shape_masked_velocity_cub[..., i] = np.where(binary_mask[..., i], norm_cub[..., i], 0)
        shape_masked_velocity_fft[..., i] = np.where(binary_mask[..., i], norm_fft[..., i], 0)


    # get vessel_wall and inside for all predictions
    bool_mask = binary_mask[...,0].astype(bool)
    vessel_wall_gt, inside_gt = split_boundary_and_interior(target_gt, bool_mask)
    vessel_wall, inside = split_boundary_and_interior(resized_target, bool_mask)
    vessel_wall_lin, inside_lin = split_boundary_and_interior(resized_lin, bool_mask)
    vessel_wall_cub, inside_cub = split_boundary_and_interior(resized_cub, bool_mask)
    vessel_wall_fft, inside_fft = split_boundary_and_interior(resized_fft, bool_mask)

    # ------------------------------------------------------------------
    # compute Divergence ∇*u - continuity equation
    # ------------------------------------------------------------------
    # for t in range(1): 
    # for t in range(resized_target.shape[-1]):
    for t in range(1):
        # define erosion of binary mask
        div_mask = binary_erosion(binary_mask[...,t], iterations=2)
        
        sr_div = compute_divergence_np(resized_target[...,t], delta_x=image_resolution, mask=div_mask)
        gt_div = compute_divergence_np(target_gt[...,t], delta_x=image_resolution, mask=div_mask)
        gtn_div = compute_divergence_np(target_gtn[...,t], delta_x=image_resolution, mask=div_mask)
        lin_div = compute_divergence_np(resized_lin[...,t], delta_x=image_resolution, mask=div_mask)
        cub_div = compute_divergence_np(resized_cub[...,t], delta_x=image_resolution, mask=div_mask)
        fft_div = compute_divergence_np(resized_fft[...,t], delta_x=image_resolution, mask=div_mask)

        # print("SR mean div   ", np.mean(np.abs(sr_div[div_mask])))
        # print("GT mean div   ", np.mean(np.abs(gt_div[div_mask])))
        # print("Lin mean div  ", np.mean(np.abs(lin_div[div_mask])))
        # print("Cub mean div  ", np.mean(np.abs(cub_div[div_mask])))
        # print("FFT mean div  ", np.mean(np.abs(fft_div[div_mask])))
        
        # Create a histogram
        div_data = [sr_div[div_mask], gt_div[div_mask], gtn_div[div_mask], lin_div[div_mask], cub_div[div_mask], fft_div[div_mask]]
        
        # Calculate Q1 and Q3 for each data set
        # q1 = [np.percentile(d, 25) for d in div_data]
        # q3 = [np.percentile(d, 75) for d in div_data]
        # q1_q3_legend = [f"{labels[i]} - Q1: {q1[i]:.0f}, Q3: {q3[i]:.0f}" for i in range(len(labels))]
        # legend_text = "\n".join(q1_q3_legend)
        
        # Print the Q1 and Q3 values for each variable
        labels = ['SR Div', 'GT Div', 'GT noise Div', 'Lin Div', 'Cub Div', 'FFT Div']
        for i in range(len(div_data)):
            mean_val = np.mean(div_data[i], dtype=np.float64)
            std_dev = np.std(div_data[i], dtype=np.float64)
            # laplacian distribution  - MAD : mean absolute deviation
            mad = np.median(np.abs(div_data[i] - np.median(div_data[i])))
            b = (1 / np.sqrt(2)) * mad
            # Update each label with mean and standard deviation
            labels[i] += r' ($\mu$={:.1f}, $\sigma$={:.1f}, b={:.1f})'.format(mean_val, std_dev, b)
    
        # Create a box plot
        width = 20
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
        
        ### plot all in one figure
        plt.figure(figsize=(10, 6))
        for i in range(len(div_data)):
            histogram, bin_edges = np.histogram(div_data[i], bins='auto') #, density=True)# 'auto')
            plt.plot(bin_edges[0:-1], histogram, label=labels[i])  # Bin edges have one extra element
            # plt.bar(bin_edges[:-1] + i * width, histogram, width=width, label=labels[i], alpha=0.5)
            
        plt.title('Divergence Values')
        plt.xlabel('Divergence [1/s]')
        plt.ylabel('Probability Density')
        plt.grid(True)
        # plt.legend(title=legend_text, loc='upper left', bbox_to_anchor=(1, 1))
        plt.legend()
        plt.tight_layout()
        plt.savefig(snr_dir + '/' + f'divergence_histogram_snr{snr}_t{t}.png', dpi=200)
        plt.show()
        
        
        ### plot subplots
        # plt.figure(figsize=(15, 10))  # Adjust figure size as needed
        # rows, cols = 2, 3
        # for i in range(len(div_data)):
        #     histogram, bin_edges = np.histogram(div_data[i], bins=100)
            
        #     plt.subplot(rows, cols, i + 1)
        #     # plt.bar(bin_edges[:-1], histogram, width=bin_edges[1] - bin_edges[0],
        #     #         color=colors[i], alpha=0.5, label=labels[i])
        #     plt.plot(bin_edges[0:-1], histogram, label=labels[i])  # Bin edges have one extra element

        #     plt.title(f'{labels[i]}')
        #     plt.xlim([-300,300])
        #     plt.ylim([0,7200])
        #     plt.xlabel('Divergence [1/s]')
        #     plt.ylabel('Count')
        #     plt.grid(True)
        # plt.suptitle('Divergence Values', fontsize=16, y=1.02)
        # plt.legend(title=legend_text, loc='upper center', bbox_to_anchor=(1.15, 1), ncol=1)
        # plt.tight_layout()
        # plt.savefig(snr_dir + '/' + f'divergence_histogram_snr{snr}_t{t}.png', dpi=200, bbox_inches='tight')
        # plt.show()

    # ------------------------------------------------------------------
    # Velocity Gradients on Vessel Wall - Wall Shear Stress (WSS)
    # ------------------------------------------------------------------
    '''
    mask = binary_mask[...,0]
    
    nx, ny, nz = target_gt.shape[1:-1]
    x = np.linspace(0, image_resolution * (nx - 1), nx)
    y = np.linspace(0, image_resolution * (ny - 1), ny)
    z = np.linspace(0, image_resolution * (nz - 1), nz)

    mx, my, mz = np.meshgrid(y, x, z) # TODO: meshgrid changes order?

    X, Y, Z = np.meshgrid(
        np.linspace(0, target_gt.shape[2], target_gt.shape[2]),
        np.linspace(0, target_gt.shape[1], target_gt.shape[1]),
        np.linspace(0, target_gt.shape[3], target_gt.shape[3])
    )

    # Compute the gradient of the mask
    grad_x, grad_y, grad_z = np.gradient(mask.astype(float))
    # Stack gradients into a normal vector array
    normals = np.stack((grad_x, grad_y, grad_z), axis=-1)
    # Compute the magnitude of the normal vectors
    norm_magnitude = np.linalg.norm(normals, axis=-1, keepdims=True)
    # Avoid division by zero in normalization
    norm_magnitude[norm_magnitude == 0] = 1
    # Normalize the normals to get unit vectors
    unit_normals = normals / norm_magnitude
    inside_normals = -unit_normals
    # Apply mask to keep only boundary normals
    vessel_wall_normals = inside_normals * np.expand_dims(mask, axis=-1)
    
    normals_x = np.where(mask, vessel_wall_normals[...,0], 0)
    normals_y = np.where(mask, vessel_wall_normals[...,1], 0)
    normals_z = np.where(mask, vessel_wall_normals[...,2], 0)

    
    points = np.vstack((X.flatten('F'), Y.flatten('F'), Z.flatten('F'))).T
    normals = np.vstack((normals_x.flatten('F'), normals_y.flatten('F'), normals_z.flatten('F'))).T

    # pv.set_jupyter_backend("trame")
    # %matplotlib widget
    pdata = pv.vector_poly_data(points, normals)
    # pdata.glyph(orient='vectors', scale=10).plot()  # scale='mag'
    plotter = pv.Plotter(notebook=True)
    glyphs = pdata.glyph(orient="vectors", factor=1)
    # plotter.add_mesh(pdata.glyph(orient="vectors", scale=5))  # Adjust scale_factor as needed
    plotter.add_mesh(glyphs)
    # plotter.show()
    plotter.show(jupyter_backend='trame')
    '''

        

    # ------------------------------------------------------------------
    # compute SNR on whole image domain on MAGNITUDE
    # ------------------------------------------------------------------
    # shape_masked_slice = shape_masked_velocity[...,z_slice, t_frame]
    # shape_masked_nonzero = shape_masked_slice[shape_masked_slice != 0]
    # sr_snr = np.mean(shape_masked_nonzero)/np.std(shape_masked_nonzero)
    
    # TODO : change computation to magnitude
    '''
    sr_snr = []
    gt_snr = []
    gtn_snr = []
    lin_snr = []
    cub_snr = []
    fft_snr = []
    # loop over time frames
    for t in range(len(masked_velocity)):
        sr_snr.append(np.mean(np.abs(masked_velocity[t])) / np.std(np.abs(nf_masked_velocity[t])))
        gt_snr.append(np.mean(np.abs(masked_velocity_gt[t])) / np.std(np.abs(nf_masked_velocity_gt[t])))
        lin_snr.append(np.mean(np.abs(masked_velocity_lin[t])) / np.std(np.abs(nf_masked_velocity_lin[t])))
        cub_snr.append(np.mean(np.abs(masked_velocity_cub[t])) / np.std(np.abs(nf_masked_velocity_cub[t])))
        fft_snr.append(np.mean(np.abs(masked_velocity_fft[t])) / np.std(np.abs(nf_masked_velocity_fft[t])))

        # sr_snr_inside = np.mean(np.abs(masked_velocity[t])) / np.std(np.abs(masked_velocity[t]))
        # gt_snr_inside = np.mean(np.abs(masked_velocity_gt[t])) / np.std(np.abs(masked_velocity_gt[t]))
        
    print("sr_snr   ", round(np.mean(sr_snr),2))
    print("gt_snr   ", round(np.mean(gt_snr),2))
    print("lin_snr  ", round(np.mean(lin_snr),2))
    print("cub_snr  ", round(np.mean(cub_snr),2))
    print("fft_snr  ", round(np.mean(fft_snr),2))

    # print("\nsr_snr inside   ", round(sr_snr_inside,2))
    # print("gt_snr inside  ", round(gt_snr_inside,2))
    
    # Open a text file in append mode to write the results
    with open(base_path + "/snr_values.txt", "a") as file:
        # Your computations of sr_snr, gt_snr, lin_snr, cub_snr, fft_snr for this specific snr       
        # Write the results into the file
        file.write(f"SNR: {snr}\n")
        file.write(f"sr_snr   : {round(np.mean(sr_snr), 2)}\n")
        file.write(f"gt_snr   : {round(np.mean(gt_snr), 2)}\n")
        file.write(f"lin_snr  : {round(np.mean(lin_snr), 2)}\n")
        file.write(f"cub_snr  : {round(np.mean(cub_snr), 2)}\n")
        file.write(f"fft_snr  : {round(np.mean(fft_snr), 2)}\n")
        file.write("\n")  # Adding a new line for clarity between SNR iterations
    print("SNR results written to vnr_results.txt")
    '''
    
    # ------------------------------------------------------------------
    # compute VNR on whole image domain on VELOCITY COMPONENTS
    # ------------------------------------------------------------------
    # shape_masked_slice = shape_masked_velocity[...,z_slice, t_frame]
    # shape_masked_nonzero = shape_masked_slice[shape_masked_slice != 0]
    # sr_snr = np.mean(shape_masked_nonzero)/np.std(shape_masked_nonzero)
    
    # dimensions = ['x', 'y', 'z']
    # for i in range(3):
    #     print("\nVNR calculation on velocity component ", dimensions[i])
    #     sr_snr = []
    #     gt_snr = []
    #     lin_snr = []
    #     cub_snr = []
    #     fft_snr = []
    #     # loop over time frames
    #     for t in range(binary_mask.shape[-1]):
    #         mask = binary_mask[...,t]
            
    #         sr_snr.append(np.mean(np.abs(resized_target[i,...,t][mask])) / np.std(np.abs(resized_target[i,...,t][~mask])))
    #         gt_snr.append(np.mean(np.abs(target_gt[i,...,t][mask])) / np.std(np.abs(target_gt[i,...,t][~mask])))
    #         lin_snr.append(np.mean(np.abs(resized_lin[i,...,t][mask])) / np.std(np.abs(resized_lin[i,...,t][~mask])))
    #         cub_snr.append(np.mean(np.abs(resized_cub[i,...,t][mask])) / np.std(np.abs(resized_cub[i,...,t][~mask])))
    #         fft_snr.append(np.mean(np.abs(resized_fft[i,...,t][mask])) / np.std(np.abs(resized_fft[i,...,t][~mask])))

    #         # sr_snr_inside = np.mean(np.abs(masked_velocity[t])) / np.std(np.abs(masked_velocity[t]))
    #         # gt_snr_inside = np.mean(np.abs(masked_velocity_gt[t])) / np.std(np.abs(masked_velocity_gt[t]))
            
    #     print("sr_snr   ", round(np.mean(sr_snr),2))
    #     print("gt_snr   ", round(np.mean(gt_snr),2))
    #     print("lin_snr  ", round(np.mean(lin_snr),2))
    #     print("cub_snr  ", round(np.mean(cub_snr),2))
    #     print("fft_snr  ", round(np.mean(fft_snr),2))

    #     # print("\nsr_snr inside   ", round(sr_snr_inside,2))
    #     # print("gt_snr inside  ", round(gt_snr_inside,2))
        
    #     # Open a text file in append mode to write the results
    #     with open(base_path + f"/vnr_values_{dimensions[i]}.txt", "a") as file:
    #         # Your computations of sr_snr, gt_snr, lin_snr, cub_snr, fft_snr for this specific snr       
    #         # Write the results into the file
    #         file.write(f"SNR: {snr}\n")
    #         file.write(f"sr_snr   : {round(np.mean(sr_snr), 2)}\n")
    #         file.write(f"gt_snr   : {round(np.mean(gt_snr), 2)}\n")
    #         file.write(f"lin_snr  : {round(np.mean(lin_snr), 2)}\n")
    #         file.write(f"cub_snr  : {round(np.mean(cub_snr), 2)}\n")
    #         file.write(f"fft_snr  : {round(np.mean(fft_snr), 2)}\n")
    #         file.write("\n")  # Adding a new line for clarity between SNR iterations
    #     print("VNR results written to vnr_results.txt")
    
    # ------------------------------------------------------------------
    # plot flow rate over frames
    # ------------------------------------------------------------------
    '''
    print("\nPlot average velocity of masked area.")
    # factor to plot in ml/s
    flow_factor = 1e6
    # compute the average velocity magnitude for each frame
    flow_rate = []
    flow_rate_gt = []
    flow_rate_gtn = []
    flow_rate_lin = []
    flow_rate_cub = []
    flow_rate_fft = []
    for i in range(shape_masked_velocity.shape[-1]): # loop over time frames
        flow_rate.append(np.sum(shape_masked_velocity[...,z_slice,i])*image_resolution**2*flow_factor)
        flow_rate_gt.append(np.sum(shape_masked_velocity_gt[...,z_slice,i])*image_resolution**2*flow_factor)
        flow_rate_gtn.append(np.sum(shape_masked_velocity_gtn[...,z_slice,i])*image_resolution**2*flow_factor)
        flow_rate_lin.append(np.sum(shape_masked_velocity_lin[...,z_slice,i])*image_resolution**2*flow_factor)
        flow_rate_cub.append(np.sum(shape_masked_velocity_cub[...,z_slice,i])*image_resolution**2*flow_factor)
        flow_rate_fft.append(np.sum(shape_masked_velocity_fft[...,z_slice,i])*image_resolution**2*flow_factor)


    # plot the average velocity over time
    plt.plot(flow_rate, label='Upsampled velocity', marker='o')
    plt.plot(flow_rate_lin, label='Linear upsampled velocity', marker='o')
    plt.plot(flow_rate_cub, label='Cubic upsampled velocity', marker='o')
    plt.plot(flow_rate_fft, label='Zerofilled velocity', marker='o')
    plt.plot(flow_rate_gt, label='Ground truth velocity', marker='o')
    plt.plot(flow_rate_gtn, label='Ground truth noise velocity', marker='o')
    # make the x label unitary range from 0 to 19
    plt.xticks(np.arange(-2, 22, 2))
    plt.xlabel('Frame')
    plt.ylabel('Flow rate [ml/s]')
    plt.legend()
    plt.title('Flow rate over time')
    plt.grid()
    plt.savefig(snr_dir + '/' + f'flow_rate_cycle_snr{snr}.png', dpi=200)
    plt.show()
    '''

    # ------------------------------------------------------------------
    # plot average velocity over frames
    # ------------------------------------------------------------------
    print("\nPlot average velocity of masked area.")
    # compute the average velocity magnitude for each frame
    avg_velocity = []
    avg_velocity_gt = []
    avg_velocity_lin = []
    avg_velocity_cub = []
    avg_velocity_fft = []
    for i in range(len(masked_velocity)):
        avg_velocity.append(np.mean(masked_velocity[i]))
        avg_velocity_gt.append(np.mean(masked_velocity_gt[i]))
        avg_velocity_lin.append(np.mean(masked_velocity_lin[i]))
        avg_velocity_cub.append(np.mean(masked_velocity_cub[i]))
        avg_velocity_fft.append(np.mean(masked_velocity_fft[i]))

    # plot the average velocity over time
    plt.plot(avg_velocity, label='Upsampled velocity', marker='o')
    plt.plot(avg_velocity_lin, label='Linear upsampled velocity', marker='o')
    plt.plot(avg_velocity_cub, label='Cubic upsampled velocity', marker='o')
    plt.plot(avg_velocity_fft, label='Zerofilled velocity', marker='o')
    plt.plot(avg_velocity_gt, label='Ground truth velocity', marker='o')
    # make the x label unitary range from 0 to 19
    plt.xticks(np.arange(-2, 22, 2))
    plt.xlabel('Frame')
    plt.ylabel('Average velocity magnitude [m/s]')
    plt.legend()
    plt.title('Average velocity magnitude over time')
    plt.grid()
    plt.savefig(snr_dir + '/' + f'velocity_magnitude_cycle_snr{snr}.png', dpi=200)
    plt.show()


    # ------------------------------------------------------------------
    # plot correlation plots of percentage of data points - inside / vessel wall
    # ------------------------------------------------------------------
    # Arrays for comparison
    correlation_names = ['Super-resolution', 'Linear', 'Cubic', 'Zerofilled']
    correlation_save_str = ['sr', 'lin', 'cub', 'fft']
    prediction_vessel_wall_list = [vessel_wall, vessel_wall_lin, vessel_wall_cub, vessel_wall_fft]
    prediction_inside_list = [inside, inside_lin, inside_cub, inside_fft]

    
    # Generate combined correlation plots for vessel wall and inside
    # for t in range(1):
    #     for i, direction in enumerate(directions):
    #         for j in range(len(prediction_vessel_wall_list)):
    #             plot_combined_correlations(gt_vessel_wall=vessel_wall_gt[i,:,t], 
    #                                        gt_inside=inside_gt[i,:,t],
    #                                        pred_vessel_wall=prediction_vessel_wall_list[j][i,:,t],
    #                                        pred_inside=prediction_inside_list[j][i,:,t],
    #                                        pred_name=correlation_names[j], direction=direction, 
    #                                        snr=snr, time=t,
    #                                        save_name=correlation_save_str[j], 
    #                                        save_path=snr_plot_dirs[j])
    
    for t in range(1):
        for j in range(len(prediction_vessel_wall_list)):
            correlation_plot = CorrelationPlotter(gt_vessel_wall=vessel_wall_gt[...,t], 
                               gt_inside=inside_gt[...,t],
                               pred_vessel_wall=prediction_vessel_wall_list[j][...,t],
                               pred_inside=prediction_inside_list[j][...,t],
                               pred_name=correlation_names[j], snr=snr, time=t,
                               save_name=correlation_save_str[j], 
                               save_path=snr_plot_dirs[j])
            correlation_plot.plot_correlation()
        
    
    # ------------------------------------------------------------------
    # plot directional error over frames on velocity magnitude
    # ------------------------------------------------------------------
    print("\nPlot directional errors of masked area.")
    directional_error_list = []
    directional_error_zero_list = []
    directional_error_linear_list = []
    directional_error_cubic_list = []
    vectorial_rmse_list = []
    rmse_list = [] 

    for i in range(num_frames):
        directional_error_list.append(directional_error(masked_velocity[i], masked_velocity_gt[i]))
        directional_error_zero_list.append(directional_error(masked_velocity_fft[i], masked_velocity_gt[i]))
        directional_error_linear_list.append(directional_error(masked_velocity_lin[i], masked_velocity_gt[i]))
        directional_error_cubic_list.append(directional_error(masked_velocity_cub[i], masked_velocity_gt[i]))
        
        vectorial_rmse_list.append(vectorial_rmse(masked_velocity[i], masked_velocity_gt[i]))
        rmse_list.append(rmse(masked_velocity[i], masked_velocity_gt[i]))

    plt.plot(directional_error_list, label='4DFlowNet')
    plt.plot(directional_error_linear_list, label='Linear interpolation')
    plt.plot(directional_error_cubic_list, label='Cubic interpolation')
    plt.plot(directional_error_zero_list, label='Zerofilled')
    # plt.text(14, 10, 'Diastolic', fontsize=20, style='normal')
    # plt.text(5, 10, 'Systolic', fontsize=20, style='normal')

    # plt.axvline(x=12, color='black', linestyle='--')
    plt.xticks(np.arange(0, 20, 2))
    plt.xlabel('Frame')
    plt.ylabel('Directional error (%)')
    plt.grid()
    plt.legend()
    plt.savefig(snr_dir + '/' + f'directional_error_cycle_snr{snr}.png', dpi=200)
    plt.show()

    # ------------------------------------------------------------------
    # crop down to mask of interest
    # ------------------------------------------------------------------
    # plot velocities
    # time frame
    t = 0

    # Summing over the first dimension (axis=0) and then finding the index of the maximum
    summed_values = np.sum(norm_gt[:, :, z_slice, t], axis=0)
    # Find the index of the maximum value in the second dimension (axis=1)
    y_slice = np.argmax(summed_values)
    # y_slice = 60 # geom3: 26, oliver: 18

    # crop to square around mask, in slice z
    crop_to_square = True

    if crop_to_square==True:
        # Define the buffer size
        buffer_size = 3  # Adjust this value as needed

        # Find the coordinates of the bounding box of the original binary mask
        x_nonzero, y_nonzero = np.nonzero(binary_mask[:,:,z_slice,t])
        if len(y_nonzero) > 0 and len(x_nonzero) > 0:
            # Get the coordinates for the bounding box
            x_min, x_max = np.min(x_nonzero), np.max(x_nonzero)
            y_min, y_max = np.min(y_nonzero), np.max(y_nonzero)

            # Calculate the width (x-axis) and height (y-axis) of the bounding box
            width = x_max - x_min
            height = y_max - y_min

            # Determine the side length of the square (max of width and height)
            side_length = max(width, height)

            # Center the square around the bounding box
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            
            # Calculate new square boundaries with buffer and ensure they are within image limits
            x_min_buffered = max(x_center - side_length // 2 - buffer_size, 0)
            x_max_buffered = min(x_center + side_length // 2 + buffer_size, binary_mask.shape[0] - 1)

            y_min_buffered = max(y_center - side_length // 2 - buffer_size, 0)
            y_max_buffered = min(y_center + side_length // 2 + buffer_size, binary_mask.shape[1] - 1)

            # Create a new binary mask with the same shape as the original
            new_binary_mask = np.zeros_like(binary_mask)

            # Set the pixels in the new binary mask within the buffered square bounding box to 1
            new_binary_mask[x_min_buffered:x_max_buffered + 1, y_min_buffered:y_max_buffered + 1, :, :] = 1


        # Find the indices where the new binary mask is True
        x_nonzero, y_nonzero = np.nonzero(np.squeeze(new_binary_mask)[:,:,z_slice,t])
        if len(y_nonzero) > 0 and len(x_nonzero) > 0:
            # Determine the bounds for cropping
            x_min, x_max = np.min(x_nonzero), np.max(x_nonzero)
            y_min, y_max = np.min(y_nonzero), np.max(y_nonzero)

            # Crop the resized_fft array using the bounding box
            crop_resized_fft = resized_fft[:, x_min:x_max + 1,y_min:y_max + 1, :, :]
            crop_resized_lin = resized_lin[:, x_min:x_max + 1,y_min:y_max + 1, :, :]
            crop_resized_cub = resized_cub[:, x_min:x_max + 1,y_min:y_max + 1, :, :]
            crop_resized_target = resized_target[:, x_min:x_max + 1,y_min:y_max + 1, :, :]
            crop_target_gt = target_gt[:, x_min:x_max + 1,y_min:y_max + 1, :, :]
        
    else:
        crop_resized_fft = resized_fft.copy()
        crop_resized_lin = resized_lin.copy()
        crop_resized_cub = resized_cub.copy()
        crop_resized_target = resized_target.copy()
        crop_target_gt = target_gt.copy()


    # ------------------------------------------------------------------
    # plot contour plots
    # ------------------------------------------------------------------
    # print("\nPlot contour plots.")

    cmap = 'viridis' # 'viridis' , 'plasma', 'hot'

    # Define the size of the subplots using gridspec_kw
    rows, cols = 3, 5
    fig, axes = plt.subplots(rows, cols) # , figsize=(20, 12))

    # Plot the third contour plot, cmap = 'viridis', 'plasma'
    for i in range(rows):
        # Calculate vmin and vmax for this row based on all images
        vmin_row = min(
            crop_resized_fft[i, :, :, z_slice, t].min(),
            crop_resized_lin[i, :, :, z_slice, t].min(),
            crop_resized_cub[i, :, :, z_slice, t].min(),
            crop_resized_target[i, :, :, z_slice, t].min(),
            crop_target_gt[i, :, :, z_slice, t].min(),
        )
        vmax_row = max(
            crop_resized_fft[i, :, :, z_slice, t].max(),
            crop_resized_lin[i, :, :, z_slice, t].max(),
            crop_resized_cub[i, :, :, z_slice, t].max(),
            crop_resized_target[i, :, :, z_slice, t].max(),
            crop_target_gt[i, :, :, z_slice, t].max(),
        )

        # Plot each image with auto-scaled vmin and vmax
        contour1 = axes[i, 0].imshow(crop_resized_fft[i, :, :, z_slice, t], cmap=cmap, vmin=vmin_row, vmax=vmax_row)
        contour2 = axes[i, 1].imshow(crop_resized_lin[i, :, :, z_slice, t], cmap=cmap, vmin=vmin_row, vmax=vmax_row)
        contour3 = axes[i, 2].imshow(crop_resized_cub[i, :, :, z_slice, t], cmap=cmap, vmin=vmin_row, vmax=vmax_row)
        contour4 = axes[i, 3].imshow(crop_resized_target[i, :, :, z_slice, t], cmap=cmap, vmin=vmin_row, vmax=vmax_row)
        contour5 = axes[i, 4].imshow(crop_target_gt[i, :, :, z_slice, t], cmap=cmap, vmin=vmin_row, vmax=vmax_row)

        # Set aspect ratio to equal for maintaining original pixel size
        for ax in axes[i]:
            ax.set_aspect('equal')  # Maintain equal aspect ratio
            # ax.set_xlim(0, np.squeeze(resized_fft)[i, :, :, z_slice].shape[1])  # Set x limits based on data shape
            # ax.set_ylim(0, np.squeeze(resized_fft)[i, :, :, z_slice].shape[0])  # Set y limits based on data shape

        # Set titles for the row, positioned on the right of the first column
        row_title = r'$v_{x}$' if i == 0 else r'$v_{y}$' if i == 1 else r'$v_{z}$'
        axes[i, 0].annotate(row_title, xy=(-0.12, 0.5), xycoords='axes fraction', fontsize=10,
                            rotation=90, ha='center', va='center')

    # Set column titles at the top
    column_titles = ['Zerofilled', 'Linear', 'Cubic', 'Superres', 'Ground Truth']
    for ax, col_title in zip(axes[0], column_titles):
        ax.set_title(col_title, fontsize=10, pad=15)

    # Remove the axis, ticks, and labels from each plot
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

    # Adjust space between plots
    plt.subplots_adjust(wspace=0.002, hspace=0.002)  # Reduce space between plots

    # # Add colorbars for each row
    # for i in range(rows):
    #     colorbar_height = 0.2
    #     cbar_ax = fig.add_axes([1.02, ((0.33-colorbar_height)/2+i*0.33), 0.02, colorbar_height])  # Adjust colorbar placement dynamically,
    #     cbar = fig.colorbar(axes[i, 0].collections[0], cax=cbar_ax)  # Use the contour from the first plot for consistent color scaling
    #     # Set colorbar ticks to be multiples of 0.1, based on current vmin and vmax
    #     # ticks = np.arange(np.floor(vmin * 10) / 10, np.ceil(vmax * 10) / 10 + 0.1, 0.1)
    #     # cbar.set_ticks(ticks)  # Set colorbar ticks to multiples of 0.1

    # Add colorbars for each row
    for i in range(rows):
        colorbar_height = 0.2
        cbar_ax = fig.add_axes([1.02, ((0.33 - colorbar_height) / 2 + i * 0.33), 0.02, colorbar_height])  # Adjust colorbar placement dynamically
        cbar = fig.colorbar(axes[i, 0].get_images()[0], cax=cbar_ax)  # Use the image object for the color scaling

    # Save the figure
    # plt.savefig(base_path + '/' + f'velocity_slice_velo_3x5_fixed_size.png', dpi=200, bbox_inches='tight')
    plt.tight_layout()
    plt.savefig(snr_dir + '/' + f'velocity_slice_3x5_snr{snr}.png', dpi=200, bbox_inches='tight')
    plt.show()


    # ------------------------------------------------------------------
    # plot velocity over line
    # ------------------------------------------------------------------
    print("\nPlot velocity over line.")

    probe_norm_gt = norm_gt[:, y_slice, z_slice, t]
    probe_norm_target = norm_target[:, y_slice, z_slice, t]
    probe_norm_lin = norm_lin[:, y_slice, z_slice, t]
    probe_norm_cub = norm_cub[:, y_slice, z_slice, t]
    probe_norm_fft = norm_fft[:, y_slice, z_slice, t]


    plt.plot(np.arange(len(probe_norm_target))*image_resolution, probe_norm_target, label='SR')
    plt.plot(np.arange(len(probe_norm_lin))*image_resolution, probe_norm_lin, label='linear')
    plt.plot(np.arange(len(probe_norm_cub))*image_resolution, probe_norm_cub, label='cubic')
    plt.plot(np.arange(len(probe_norm_fft))*image_resolution, probe_norm_fft, label='zerofill')
    plt.plot(np.arange(len(probe_norm_gt))*image_resolution, probe_norm_gt, label='HR (GT)')

    plt.ylabel('velocity [m/s]')
    plt.xlabel('position')
    plt.legend()
    plt.savefig(snr_dir + '/' + f'velocity_probe_w_snr{snr}.png', dpi=200)
    plt.show()


    # ------------------------------------------------------------------
    print(f'\n*** Analysis for SNR {snr} finished ! ***\n')
    
    
    