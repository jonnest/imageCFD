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

# folder = 'geom3'
# base_path = f'/Users/jonathan/Documents/applications/superres/{folder}/time_data/'
base_path = '/Users/jonathan/Documents/applications/superres/ubend_1000_m1/vtk_eval'
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