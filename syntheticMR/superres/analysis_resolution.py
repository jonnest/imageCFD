''' Resolution of 4DFlowNet results

- compare HR, inference
- compare LR with same upsampled matrix size !
- linearly interpolated
- compare to b-spline interpolation

TODO 
- for interpolated_data_exists = True, do also u_linnf interpolation 
- number of bins in histogram
- histogram as density function??

'''

import h5py
import pyvista as pv
import numpy as np 
import matplotlib.pyplot as plt 
import sys,os,glob,re
from scipy.ndimage import zoom as zoom
from skimage.transform import resize
from IPython.display import Image as ipImage
from scipy.ndimage import binary_erosion
from scipy.stats import skew, kurtosis
from scipy.signal import savgol_filter
from scipy.stats import linregress

from gridOperators import *
from filters import *
from fileOperators import *
from functionsRST import *
import h5_utils as h5utils
from plot_functions import *
from fourierOperators import * 
from computeMetrics import *

folder = 'geom3'
base_path = f'/Users/jonathan/Documents/applications/superres/{folder}/time_data/'

# targetSNR 
snr_list = [20]

image_resolution = 0.4e-3
reduction_factor = 0.5

num_frames = 20

interpolated_data_exists = False # True
read_only_two = True

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
name_list = ['super-resolution', 'linear', 'cubic', 'zerofilled', 'gt']
save_string = ['sr', 'lin', 'cub', 'fft', 'gt']

# loop over available SNR evaluations
for snr in snr_list:
    print(f'\n*** Start analysis for SNR {snr} ! ***\n')

    lr_nf_folder = f'lr_data'
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
    gt_dir = os.path.join(snr_dir, f'plots_gt_snr{snr}')
    gtn_dir = os.path.join(snr_dir, f'plots_gtn_snr{snr}')
    lin_dir = os.path.join(snr_dir, f'plots_lin_snr{snr}')
    cub_dir = os.path.join(snr_dir, f'plots_cub_snr{snr}')
    fft_dir = os.path.join(snr_dir, f'plots_fft_snr{snr}')
    
    lin_nf_dir = os.path.join(snr_dir, f'plots_lin_nf')
    
    snr_plot_dirs = [sr_dir, lin_dir, cub_dir, fft_dir]
    
    os.makedirs(sr_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(gtn_dir, exist_ok=True)
    os.makedirs(lin_dir, exist_ok=True)
    os.makedirs(cub_dir, exist_ok=True)
    os.makedirs(fft_dir, exist_ok=True)
    
    os.makedirs(lin_nf_dir, exist_ok=True)
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

    # read LR noise free data
    for volume_index,file in enumerate(fetchFiles(os.path.join(base_path,lr_nf_folder) + '/*',[lr_string], ['.vtk'])):
        
        # print("\nvolume_index  ", volume_index)
        # Open the current h5 file in read mode.
        data = h5py.File(file,'r')
        # print("LR file ", target.split('/')[-1] )
        case_name = os.path.splitext(os.path.basename(file))[0]
        # If this is the first file, initialize the data arrays by loading and squeezing
        # single data volumes from the file. `np.newaxis` is used to add a new axis,
        # preparing for concatenation along this axis for subsequent files.
        if volume_index == 0:
            ulnf = np.squeeze(data['u'][...])[...,np.newaxis]
            vlnf = np.squeeze(data['v'][...])[...,np.newaxis]
            wlnf = np.squeeze(data['w'][...])[...,np.newaxis]

            if interpolated_data_exists==False:
                # linearly interpolated data
                u_linnf = interpolate_3d_grid(ulnf[...,volume_index], mult_factor=2, method='linear',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis]
                v_linnf = interpolate_3d_grid(vlnf[...,volume_index], mult_factor=2, method='linear',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis]
                w_linnf = interpolate_3d_grid(wlnf[...,volume_index], mult_factor=2, method='linear',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis]

        else:
            # For subsequent files, concatenate the new data volumes along the last axis.
            ulnf = np.concatenate((ulnf,np.squeeze(data['u'][...])[...,np.newaxis]),axis=-1)
            vlnf = np.concatenate((vlnf,np.squeeze(data['v'][...])[...,np.newaxis]),axis=-1)
            wlnf = np.concatenate((wlnf,np.squeeze(data['w'][...])[...,np.newaxis]),axis=-1) 
                
            if interpolated_data_exists==False:
                # linearly interpolated data
                u_linnf = np.concatenate((u_linnf,interpolate_3d_grid(ulnf[...,volume_index], mult_factor=2, method='linear',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis]),axis=-1) 
                v_linnf = np.concatenate((v_linnf,interpolate_3d_grid(vlnf[...,volume_index], mult_factor=2, method='linear',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis]),axis=-1) 
                w_linnf = np.concatenate((w_linnf,interpolate_3d_grid(wlnf[...,volume_index], mult_factor=2, method='linear',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis]),axis=-1) 
            
            if read_only_two:
                if volume_index==1:
                    break
        data.close()   
    print("Read LR noisefree files, nr      ", volume_index)
    
    
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
            # mask = np.concatenate((mask,np.squeeze(data['mask'][...])[...,np.newaxis]),axis=-1)
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
            # mask = np.concatenate((mask,np.squeeze(data['mask'][...])[...,np.newaxis]),axis=-1)
            if read_only_two:
                if volume_index==1:
                    break
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

    u_linnf = u_linnf[np.newaxis,...]
    v_linnf = v_linnf[np.newaxis,...]
    w_linnf = w_linnf[np.newaxis,...]
    
    # concatenate x,y,z in second dimension
    target = np.concatenate((us,vs,ws),axis=0)
    target_gt = np.concatenate((ugt,vgt,wgt),axis=0)
    target_gtn = np.concatenate((ugtn,vgtn,wgtn),axis=0)
    lin = np.concatenate((u_lin,v_lin,w_lin),axis=0)
    cub = np.concatenate((u_cub,v_cub,w_cub),axis=0)
    fft = np.concatenate((u_fft,v_fft,w_fft),axis=0)
    linnf = np.concatenate((u_linnf,v_linnf,w_linnf),axis=0)
    
    # resize like this but for every frame in the dataset separately
    resized_target = np.zeros(target_gt.shape)
    resized_lin = np.zeros(target_gt.shape)
    resized_cub = np.zeros(target_gt.shape)
    resized_fft = np.zeros(target_gt.shape)
    resized_linnf = np.zeros(target_gt.shape)
    
    for i in range(target.shape[-1]):
        resized_target[...,i] = resize(target[...,i], target_gt[...,i].shape, order=3, mode='constant', anti_aliasing=True)
        resized_lin[...,i] = resize(lin[...,i], target_gt[...,i].shape, order=3, mode='constant', anti_aliasing=True)
        resized_cub[...,i] = resize(cub[...,i], target_gt[...,i].shape, order=3, mode='constant', anti_aliasing=True)
        resized_fft[...,i] = resize(fft[...,i], target_gt[...,i].shape, order=3, mode='constant', anti_aliasing=True)
        resized_linnf[...,i] = resize(linnf[...,i], target_gt[...,i].shape, order=3, mode='constant', anti_aliasing=True)
        
    # set non-fluid field to zero
    target_gt = np.where(mask, target_gt[:,..., :], 0)
    target_gtn = np.where(mask, target_gtn[:,..., :], 0)
    resized_target = np.where(mask, resized_target[:,..., :], 0)
    resized_lin = np.where(mask, resized_lin[:,..., :], 0)
    resized_cub = np.where(mask, resized_cub[:,..., :], 0)
    resized_fft = np.where(mask, resized_fft[:,..., :], 0)
    resized_linnf = np.where(mask, resized_linnf[:,..., :], 0)
    
    # ------------------------------------------------------------------
    # calculate resulting SNR
    # ------------------------------------------------------------------
    # snr for time frame 0 and direction x
    bool_mask = np.squeeze(mask).astype(bool)
    cub_snr = np.mean(resized_cub[0,bool_mask,0])/np.std(target_gt[0,bool_mask,0]-resized_cub[0,bool_mask,0])

    cub_snr_abs = np.mean(np.abs(resized_cub[0,bool_mask,0]))/np.std(resized_cub[0,~bool_mask,0])

    print("cub snr        ", cub_snr)
    print("cub snr_abs    ", cub_snr_abs)

    arrays = [resized_target, resized_lin, resized_linnf, resized_cub, resized_fft]
    array_names = ['sr', 'lin noise', 'lin nf', 'cub noise', 'zerofilled']
    
    for a in range(len(arrays)):
        array = arrays[a]
        # Calculate signal energy of the ground truth in the masked region
        signal = np.mean(array[0,bool_mask,0] ** 2)

        # Calculate noise as the difference between ground truth and SR output in the masked region
        noise = np.mean((target_gtn[0,bool_mask,0] - array[0,bool_mask,0]) ** 2)

        # Compute SNR in dB
        snr_db = 10 * np.log10(signal / noise)
        snr_direct = signal / noise
        print(f"\n{array_names[a]} :")
        print("snr db      ", snr_db)
        print("snr direct  ", snr_direct)
    
    
    # ------------------------------------------------------------------
    # transform to Fourier domain
    # ------------------------------------------------------------------
   
    kwindow = "hybrid"
    mult_factor = 1/2    
    axes = (0,1,2)
    bin_size = 10
    dimensions = ['x', 'y', 'z']

    # effect of noise
    noise_hr_u = np.load(base_path + f"/noise_hr_u_snr{snr}.npy")
    noise_hr_v = np.load(base_path + f"/noise_hr_v_snr{snr}.npy")
    noise_hr_w = np.load(base_path + f"/noise_hr_w_snr{snr}.npy")

    noise_lr_u = np.load(base_path + f"/noise_lr_u_snr{snr}.npy")
    noise_lr_v = np.load(base_path + f"/noise_lr_v_snr{snr}.npy")
    noise_lr_w = np.load(base_path + f"/noise_lr_w_snr{snr}.npy")
    
    noise_lin_u = np.squeeze(interpolate_3d_grid(noise_lr_u, mult_factor=2, method='linear',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis])
    
    # un_fft = np.fft.fftshift(np.fft.fftn(noise_u,axes=axes,norm=norm),axes=axes)
    # un_hrfft_shape, un_hrfft = hr_crop_box(noise_u)
    
    # plot noise
    bin_size = 1
    norm = 'backward' # 'ortho'
    '''
    ### Effect of NORM - always use same for fft and ifft
    ugtn_fft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(target_gtn[i,...,t]), norm=norm)
    # plot_histogram(target_gtn[i,...,t].ravel(), 'test')

    print("\nInverse")
    target_gtn2 = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(ugtn_fft), norm=norm))
    # plot_histogram(np.real(target_gtn2).ravel(), 'test')

    histogram1, bin_edges1 = np.histogram(target_gtn[i,...,t].ravel(), bins=100)
    histogram2, bin_edges2 = np.histogram(np.real(target_gtn2).ravel(), bins=100)
    # Plot the histogram of frequency magnitudes
    plt.figure(figsize=(8, 6))
    plt.plot(bin_edges1[0:-1], histogram1)
    plt.plot(bin_edges2[0:-1], histogram2)
    plt.show()
    '''
    plot_log_square_fourier_norm_colored(noise_hr_u, bin_size, snr, time_frame=0, dim='hr_u', name_string='noise_hr_u',target=gtn_dir)
    # plot_log_square_fourier_norm_colored(noise_lr_u, bin_size, snr, time_frame=0, dim='lr_u', name_string='noise_lr_u',target=gtn_dir)
    plot_log_square_fourier_norm_colored(noise_lin_u, bin_size, snr, time_frame=0, dim='lin_u', name_string='noise_lr_u',target=gtn_dir)
    # compare k-space
    # noise_hr_u_fft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(noise_hr_u),axes=axes,norm=norm),axes=axes)
    noise_hr_u_hrfft_shape, noise_hr_u_hrfft = hr_crop_box(noise_hr_u)
    
    # plot_2x2_histograms(noise_hr_u_hrfft, snr, 'only_noise_hr_u', target=gtn_dir)
    # plot_2x2_histograms(noise_hr_u, snr, 'noise_hr_u', target=gtn_dir)
    # plot_2x2_histograms(noise_lr_u, snr, 'noise_lr_u', target=gtn_dir)
    # plot_2x2_histograms(noise_lin_u, snr, 'noise_lin_u', target=gtn_dir)
    
    # # hr k-space
    # noise_u_hrfft_crop, noise_u_hrfft = hr_crop_box(noise_u)
    # plot_2x2_histograms(noise_u_hrfft, snr, 'noise_hr_u', target=gtn_dir)
    
    # # lr k-space
    # noise_u_lrfft_crop, noise_u_lrfft = lr_crop_box(noise_u)
    # plot_2x2_histograms(noise_u_lrfft, snr, 'noise_lr_u', target=gtn_dir)

    hr_x_lims = [-25,25]
    lr_x_lims = [-100,100]
    for t in range(1):
        #for t in range(us.shape[-1]):
        for i in range(1):
            i = 1 # only for getting y cooridnate
            
            ugtn_fft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(target_gtn[i,...,t]),axes=axes,norm=norm),axes=axes)            
            ugtn_hrfft_shape, ugtn_hrfft = hr_crop_box(ugtn_fft)            
            ugtn_lrfft_shape, ugtn_lrfft = lr_crop_box(ugtn_fft)            
            
            ugt_fft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(target_gt[i,...,t]),axes=axes,norm=norm),axes=axes)
            ugt_hrfft_shape, ugt_hrfft = hr_crop_box(ugt_fft)
            ugt_lrfft_shape, ugt_lrfft = lr_crop_box(ugt_fft)
                 
            us_fft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(resized_target[i,...,t]),axes=axes,norm=norm),axes=axes)
            us_hrfft_shape, us_hrfft = hr_crop_box(us_fft)
            us_lrfft_shape, us_lrfft = lr_crop_box(us_fft)
        
            ulin_fft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(resized_lin[i,...,t]),axes=axes,norm=norm),axes=axes)
            ulin_hrfft_shape, ulin_hrfft = hr_crop_box(ulin_fft)
            ulin_lrfft_shape, ulin_lrfft = lr_crop_box(ulin_fft)

            ulinnf_fft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(resized_linnf[i,...,t]),axes=axes,norm=norm),axes=axes)
            ulinnf_hrfft_shape, ulinnf_hrfft = hr_crop_box(ulinnf_fft)
            ulinnf_lrfft_shape, ulinnf_lrfft = lr_crop_box(ulinnf_fft)
            
            ucub_fft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(resized_cub[i,...,t]),axes=axes,norm=norm),axes=axes)
            ucub_hrfft_shape, ucub_hrfft = hr_crop_box(ucub_fft)
            ucub_lrfft_shape, ucub_lrfft = lr_crop_box(ucub_fft)

            # extract zerofilled kspace
            ufft_fft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(resized_fft[i,...,t]),axes=axes,norm=norm),axes=axes)
            ufft_lrfft_shape, ufft_lrfft = lr_crop_box(ufft_fft)

           ### Fourier Shell Correlation
            array_shape_list = [us_fft, ulin_fft, ulinnf_fft, ucub_fft, ufft_fft, ugtn_fft]
            name_list = ['sr', 'lin noise', 'lin noisefree', 'cub', 'zerofilled', 'gt noise']
            for j in range(len(array_shape_list)):
                frc, _, max_radius = compute_fsc(ugt_fft, array_shape_list[j])
                plt.plot(range(len(frc)), frc, label=name_list[j])
            plt.axvline(x=max_radius, color="grey", linestyle=":", label="full sphere")
            plt.legend()
            plt.xlabel("Radial Frequency Shell")
            plt.ylabel("Fourier Shell Correlation (FSC)")
            plt.title("Fourier Shell Correlation")
            plt.savefig(snr_dir + '/' + f'fourier_ring_corr_{snr}_t{t}_{dimensions[i]}.png', dpi=200, bbox_inches='tight')
            plt.show()
                        
            ### plot fourier space energy
            # plot_log_square_fourier_norm_colored(ugtn_fft, bin_size, snr, time_frame=t, dim=dimensions[i], name_string='gt noise',target=gtn_dir)
            # plot_log_square_fourier_norm_colored(ugt_fft, bin_size, snr, time_frame=t, dim=dimensions[i], name_string='gt',target=gt_dir)
            # plot_log_square_fourier_norm_colored(us_fft, bin_size, snr, time_frame=t, dim=dimensions[i], name_string='sr',target=sr_dir)
            # plot_log_square_fourier_norm_colored(ulin_fft, bin_size, snr, time_frame=t, dim=dimensions[i], name_string='lin',target=lin_dir)
            # plot_log_square_fourier_norm_colored(ulinnf_fft, bin_size, snr, time_frame=t, dim=dimensions[i], name_string='lin_nf',target=lin_nf_dir)
            # plot_log_square_fourier_norm_colored(ucub_fft, bin_size, snr, time_frame=t, dim=dimensions[i], name_string='cub',target=cub_dir)
            
            ### plot fourier space energy correlation to ground truth
            plot_fourier_correlation_kspace_split(ugt_fft, ugtn_fft, snr, time_frame=t, dim=dimensions[i], name_string='gtn',target=gtn_dir)
            plot_fourier_correlation_kspace_split(ugt_fft, us_fft, snr, time_frame=t, dim=dimensions[i], name_string='sr',target=sr_dir)
            plot_fourier_correlation_kspace_split(ugt_fft, ulin_fft, snr, time_frame=t, dim=dimensions[i], name_string='lin',target=lin_dir)
            plot_fourier_correlation_kspace_split(ugt_fft, ucub_fft, snr, time_frame=t, dim=dimensions[i], name_string='cub',target=cub_dir)
            
            # plot_fourier_correlation_kspace_colored(ugt_fft, ugtn_fft, snr, time_frame=t, dim=dimensions[i], name_string='gtn',target=gtn_dir)
            # plot_fourier_correlation_kspace_colored(ugt_fft, us_fft, snr, time_frame=t, dim=dimensions[i], name_string='sr',target=sr_dir)
            # plot_fourier_correlation_kspace_colored(ugt_fft, ulin_fft, snr, time_frame=t, dim=dimensions[i], name_string='lin',target=lin_dir)
            # plot_fourier_correlation_kspace_colored(ugt_fft, ucub_fft, snr, time_frame=t, dim=dimensions[i], name_string='cub',target=cub_dir)



            ### plot k-space statistics of low and high resolution space
            # lrfft_list = [us_lrfft, ulin_lrfft, ulinnf_lrfft, ugtn_lrfft, ugt_lrfft]
            # hrfft_list = [us_hrfft, ulin_hrfft, ulinnf_hrfft, ugtn_hrfft, ugt_hrfft]
            # name_list = ['sr', 'lin noise', 'lin nf', 'gt noise', 'gt']
            lrfft_list = [us_lrfft, ulin_lrfft, ucub_lrfft, ugtn_lrfft, ugt_lrfft]
            hrfft_list = [us_hrfft, ulin_hrfft, ucub_hrfft, ugtn_hrfft, ugt_hrfft]
            name_list = ['sr', 'lin noise', 'cub noise', 'gt noise', 'gt']
            print("\nLow resolution histogram")            
            plot_2x2_histograms_list(lrfft_list, lr_x_lims, snr, t, dimensions[i], name_list, snr_dir, '_lr')           
            print("\nHigh resolution histogram")
            plot_2x2_histograms_list(hrfft_list, hr_x_lims, snr, t, dimensions[i], name_list, snr_dir, '_hr')

            ### plot k-space statistics of difference of low and high resolution space to GT
            # lrfft_diff_list = [ugt_lrfft-us_lrfft, ugt_lrfft-ulin_lrfft, ugt_lrfft-ulinnf_lrfft, ugt_lrfft-ugtn_lrfft]
            # hrfft_diff_list = [ugt_hrfft-us_hrfft, ugt_hrfft-ulin_hrfft, ugt_hrfft-ulinnf_hrfft, ugt_hrfft-ugtn_hrfft]
            # name_list_diff = ['sr', 'lin noise', 'lin nf', 'gt noise']
            lrfft_diff_list = [ugt_lrfft-us_lrfft, ugt_lrfft-ulin_lrfft, ugt_lrfft-ucub_lrfft, ugt_lrfft-ugtn_lrfft]
            hrfft_diff_list = [ugt_hrfft-us_hrfft, ugt_hrfft-ulin_hrfft, ugt_hrfft-ucub_hrfft, ugt_hrfft-ugtn_hrfft]
            name_list_diff = ['sr', 'lin noise', 'cub noise', 'gt noise']            
            print("\nLow resolution difference to GT histogram")  
            plot_2x2_histograms_list(lrfft_diff_list, lr_x_lims, snr, t, dimensions[i], name_list_diff, snr_dir, '_lr_diff_gt')
            print("\nHigh resolution difference to GT histogram")
            plot_2x2_histograms_list(hrfft_diff_list, hr_x_lims, snr, t, dimensions[i], name_list_diff, snr_dir, '_hr_diff_gt')
            
            ### absolute error    
            mae_gtn = mean_absolute_error_complex(ugtn_hrfft, ugt_hrfft)
            mae_us = mean_absolute_error_complex(us_hrfft, ugt_hrfft)
            mae_ulin = mean_absolute_error_complex(ulin_hrfft, ugt_hrfft)
            mae_ulinnf = mean_absolute_error_complex(ulinnf_hrfft, ugt_hrfft)
            mae_ucub = mean_absolute_error_complex(ucub_hrfft, ugt_hrfft)
            
            print("\nMean absolute error")
            print("mae_gtn    ", mae_gtn)
            print("mae_us     ", mae_us)
            print("mae_lin    ", mae_ulin)
            print("mae_linnf  ", mae_ulinnf)
            print("mae_cub    ", mae_ucub)   
            
            ### relative error        
            mre_gtn = mean_relative_error_complex(ugtn_hrfft, ugt_hrfft)
            mre_us = mean_relative_error_complex(us_hrfft, ugt_hrfft)
            mre_ulin = mean_relative_error_complex(ulin_hrfft, ugt_hrfft)
            mre_ulinnf = mean_relative_error_complex(ulinnf_hrfft, ugt_hrfft)
            mre_ucub = mean_relative_error_complex(ucub_hrfft, ugt_hrfft)

            print("\nMean relative error")
            print("mre_gtn     ", mre_gtn)
            print("mre_us      ", mre_us)
            print("mre_ulin    ", mre_ulin)
            print("mre_ulinnf  ", mre_ulinnf)
            print("mre_ucub    ", mre_ucub)            
            
            
            
            
            