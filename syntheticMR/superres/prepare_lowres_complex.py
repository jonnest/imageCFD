''' Ferdian 2020 4DFlowNet code
author: Oliver Wieben
'''

import numpy as np
import os
import h5py
import random
import matplotlib.pyplot as plt
# import preprocessing as pre
import time
import fft_downsampling as fft
import h5_utils as h5utils
import scipy.ndimage as ndimage
import pandas as pd
import pyvista as pv
    
from fourierOperators import * 
from flownet_functions import *
from plot_functions import *

if __name__ == '__main__':
    # NOTE: input is velocity image !!!
    # Update your path here
    folder = 'geom3'
   
    base_path = f'/Users/jonathan/Documents/applications/superres/{folder}/time_data/' 

    template_filepath = base_path + 'mag_template_aligned.h5'   
    
    # Change your case name, template idx (see mag_template_aligned.h5), and target SNR here
    # case_name = 'healthy-05mm3_sliced'
    case_name = 'data_HR_r4'
    case_name_noise = 'data_HR_noise'
    new_case_name_nf = 'data_LR'
    
    resolution = 0.4e-3    
    template_idx = 1
    
    # Set the signal-to-noise ratio (SNR) and the reduction factor for downsampling.
    snr = 20
    
    # Calculate the encoding velocity (venc) with an extra margin and rounding to the nearest 0.5.
    venc = 1.0
        
    new_case_name = case_name.replace('_HR_', '_LR_')
    
    # make output dir
    output_dir = os.path.join(base_path, f'lr_data_snr{snr}')
    os.makedirs(output_dir, exist_ok=True)
    
    output_dir_noise = os.path.join(base_path, f'hr_data_snr{snr}')
    os.makedirs(output_dir_noise, exist_ok=True)

    output_dir_nf = os.path.join(base_path, f'lr_data')
    os.makedirs(output_dir_nf, exist_ok=True)
    
    input_filepath  = f'{base_path}/{case_name}.h5'

    downsample = 2
    crop_ratio = 1 / downsample
    is_mask_saved = False 

    # ------------------------------------------------------------------
    # Define the MRI encoding directions as unit vectors along the x, y, and z axes.
    enc_dirs = np.array([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]])

    # Calculate the encoding strength (kv_rec) based on the encoding velocity (venc).
    # It starts with 0 (no encoding) and then uses the formula π/venc for each direction.
    kv_rec = np.concatenate(([0], np.pi / np.array([venc])))
    kv_rec = kv_rec[:, None]  # Add a new axis to make kv_rec a column vector.
    kv_rec = np.tile(kv_rec, (1, len(enc_dirs)))  # Repeat kv_rec for each encoding direction.

    # ------------------------------------------------------------------
    snr_u_list = []
    # Load the magnitude template
    # load template
    with h5py.File(template_filepath, 'r') as hf:
        template = np.asarray(hf.get('mag_template')[template_idx])
        vessel = np.asarray(hf.get('vessel')[template_idx])
        template_mask = np.asarray(hf.get('mask')[template_idx])

    print("template       ", template.min(), template.max(), template.mean())
    print("vessel         ", vessel.min(), vessel.max(), vessel.mean())
    print("template_mask  ", template_mask.min(), template_mask.max(), template_mask.mean())

    # Load the mask once
    with h5py.File(input_filepath, mode = 'r' ) as hf:
        # TODO
        dx = resolution # np.asarray(hf['dx'])
        data_count = len(hf.get("u"))
        print('Data count ', data_count)
        case_mask = np.asarray(hf.get('mask'))
        if len(case_mask.shape) == 4:
            case_mask = case_mask[0,...]

    # Create the synthetic magnitude based on template and case_mask
    print("Preparing magnitude from template...")
    mag_image_org = prepare_magnitude(template, vessel, template_mask, case_mask)
    
    pad_x, pad_y, pad_z = calculate_pad(mag_image_org.shape, downsample)

    # Create the magnitude based on the possible values
    ## This is a part of augmentation to make sure we have varying magnitude    
    
    mag_image = pad(mag_image_org, pad_x, pad_y, pad_z)
   
    start_time = time.time()
    # number of time frames (frames, x, y, z)
    for idx in range(data_count):
        
        print(f"\nProcessing {idx+1}/{data_count} - SNR {snr}")
        
        # Load the velocity U V W from H5
        with h5py.File(input_filepath, mode = 'r' ) as hf:
            # print(np.asarray(hf['u']))
            hr_u = np.asarray(hf['u'][idx])
            hr_v = np.asarray(hf['v'][idx])
            hr_w = np.asarray(hf['w'][idx])
            
            # check the max velocity per component (add 10% ceiling to avoid aliasing)
            # max_vel = np.max((max_u, max_v, max_w)) * 1.1
            # randindx = np.random.randint(2) # pick the first or second lowest venc above max vel
            # venc = venc_values[np.where(venc_values > max_vel)][randindx]
            venc = 1.0

            # TODO : if error, comment padding
            hr_u = pad(hr_u, pad_x, pad_y, pad_z)
            hr_v = pad(hr_v, pad_x, pad_y, pad_z) 
            hr_w = pad(hr_w, pad_x, pad_y, pad_z)
            
        # downsampling and adding complex noise in kspace
        # lr_u, mag_u, lr_u_, mag_u_, noise_lr_u = downsample_complex_noise_hr(hr_u, mag_image, venc, crop_ratio, snr)
        # lr_v, mag_v, lr_v_, mag_v_, noise_lr_v = downsample_complex_noise_hr(hr_v, mag_image, venc, crop_ratio, snr)
        # lr_w, mag_w, lr_w_, mag_w_, noise_lr_w = downsample_complex_noise_hr(hr_w, mag_image, venc, crop_ratio, snr)
        
        lr_u, mag_u, lr_u_, mag_u_, noise_lr_u = downsample_complex_noise_lr(hr_u, mag_image, venc, crop_ratio, snr)
        lr_v, mag_v, lr_v_, mag_v_, noise_lr_v = downsample_complex_noise_lr(hr_v, mag_image, venc, crop_ratio, snr)
        lr_w, mag_w, lr_w_, mag_w_, noise_lr_w = downsample_complex_noise_lr(hr_w, mag_image, venc, crop_ratio, snr)

        # # add noise to high resolution image
        hr_u_noise, hr_mag_u, noise_hr_u = add_complex_noise(hr_u, mag_image, venc, snr)
        hr_v_noise, hr_mag_v, noise_hr_v = add_complex_noise(hr_v, mag_image, venc, snr)
        hr_w_noise, hr_mag_w, noise_hr_w = add_complex_noise(hr_w, mag_image, venc, snr)

        # unpad arrays
        lr_u = unpad(lr_u, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        lr_v = unpad(lr_v, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        lr_w = unpad(lr_w, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        mag_u = unpad(mag_u, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        mag_v = unpad(mag_v, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        mag_w = unpad(mag_w, pad_x//downsample, pad_y//downsample, pad_z//downsample)

        lr_u_ = unpad(lr_u_, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        lr_v_ = unpad(lr_v_, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        lr_w_ = unpad(lr_w_, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        mag_u_ = unpad(mag_u_, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        mag_v_ = unpad(mag_v_, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        mag_w_ = unpad(mag_w_, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        
        hr_u_noise = unpad(hr_u_noise, pad_x , pad_y , pad_z )
        hr_v_noise = unpad(hr_v_noise, pad_x , pad_y , pad_z )
        hr_w_noise = unpad(hr_w_noise, pad_x , pad_y , pad_z )
        hr_mag_u = unpad(hr_mag_u, pad_x , pad_y , pad_z )
        hr_mag_v = unpad(hr_mag_v, pad_x , pad_y , pad_z )
        hr_mag_w = unpad(hr_mag_w, pad_x , pad_y , pad_z )

        hr_u = unpad(hr_u, pad_x, pad_y , pad_z )
        hr_v = unpad(hr_v, pad_x, pad_y , pad_z )
        hr_w = unpad(hr_w, pad_x, pad_y , pad_z )

        # plot noise
        bin_size = 5
        
        if idx==0:
            # plot_2x2_histograms(noise_hr_u, snr, 'noise_hr_x', target=base_path)
            # plot_2x2_histograms(noise_lr_u, snr, 'noise_lr_x', target=base_path)
            plot_log_square_fourier_norm_colored(noise_hr_u, bin_size, snr, time_frame=0, dim='hr_u', name_string='noise_hr_u', target=base_path)
            plot_log_square_fourier_norm_colored(noise_lr_u, bin_size, snr, time_frame=0, dim='lr_u', name_string='noise_lr_u', target=base_path)
        
        
        # unpad noise
        noise_hr_u = unpad(noise_hr_u, pad_x , pad_y , pad_z )
        noise_hr_v = unpad(noise_hr_v, pad_x , pad_y , pad_z )
        noise_hr_w = unpad(noise_hr_w, pad_x , pad_y , pad_z )
        noise_lr_u = unpad(noise_lr_u, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        noise_lr_v = unpad(noise_lr_v, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        noise_lr_w = unpad(noise_lr_w, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        
        # save noise
        if idx==0:
            np.save(base_path + f'/noise_hr_u_snr{snr}.npy', noise_hr_u)
            np.save(base_path + f'/noise_hr_v_snr{snr}.npy', noise_hr_v)
            np.save(base_path + f'/noise_hr_w_snr{snr}.npy', noise_hr_w)

            np.save(base_path + f'/noise_lr_u_snr{snr}.npy', noise_lr_u)
            np.save(base_path + f'/noise_lr_v_snr{snr}.npy', noise_lr_v)
            np.save(base_path + f'/noise_lr_w_snr{snr}.npy', noise_lr_w)
        
        mask_image = pad(case_mask, pad_x, pad_y, pad_z)  ## OWO added
        mask_image = mask_image.astype(np.float32)
        mask_image = ndimage.zoom(mask_image, crop_ratio, order=1) ## OWO added
        mask_image = unpad(mask_image, pad_x//downsample, pad_y//downsample, pad_z//downsample) ## OWO added
        
        ### verify the SNR       
        # mask image
        binary_mask = np.zeros(mask_image.shape,dtype=bool)
        binary_mask[mask_image>0] = True
        hr_binary_mask = case_mask.astype(bool)
        
        # invert binary mask - no fluid 
        nofluid_mask = ~binary_mask
    
        # complex images
        img_u = mag_u * np.exp(1j*lr_u)
        img_v = mag_v * np.exp(1j*lr_v)
        img_w = mag_w * np.exp(1j*lr_w)
        # image no noise 
        img_u_ = mag_u_ * np.exp(1j*lr_u_)
        img_v_ = mag_v_ * np.exp(1j*lr_v_)
        img_w_ = mag_w_ * np.exp(1j*lr_w_)
                
        ### compute SNR 
        ### np.mean(np.abs(S)) / np.std(np.abs(S)-np.abs(S_noisy)))
        hr_snr_org = np.max(np.abs(mag_image_org[hr_binary_mask]))/np.std(np.abs(mag_image_org[~hr_binary_mask]))         

        snr_u = np.mean(np.abs(img_u))/np.std(np.abs(img_u)-np.abs(img_u_))

        hr_snr_u = np.mean(np.abs(hr_mag_u))/np.std(np.abs(hr_mag_u)-np.abs(mag_image_org))
             
        print("hr_snr_org (original image)   ", round(hr_snr_org,1))
        print("snr_u                         ", round(snr_u,1))
        print("HR snr_w                      ", round(hr_snr_u,1))
        snr_u_list.append(snr_u)
        
        
        ### plot slice
        z_slice = 14
        # low resolution noise 
        plt.imshow(lr_w[...,z_slice], cmap='viridis', interpolation='none')
        plt.colorbar(label='Velocity [m/s]')
        plt.show()
        # high resolution added noise 
        plt.imshow(hr_w_noise[...,int(z_slice*2)], cmap='viridis', interpolation='none')
        plt.colorbar(label='Velocity [m/s]')
        plt.show()
        # low resolution noise         
        plt.imshow(mag_w[...,z_slice], cmap='viridis', interpolation='none')
        plt.colorbar(label='LR Magnitude')
        plt.show()
        # original magnitude
        plt.imshow(mag_image[...,int(z_slice*2)], cmap='viridis', interpolation='none')
        plt.colorbar(label='Origianl Magnitude')
        plt.show()

        
        
        if idx == 0:
            u = np.squeeze(lr_u)[np.newaxis,...]
            v = np.squeeze(lr_v)[np.newaxis,...]
            w = np.squeeze(lr_w)[np.newaxis,...]

        else:
            # For subsequent files, concatenate the new data volumes along the last axis.
            u = np.concatenate((u,np.squeeze(lr_u)[np.newaxis,...]),axis=0)
            v = np.concatenate((v,np.squeeze(lr_v)[np.newaxis,...]),axis=0)
            w = np.concatenate((w,np.squeeze(lr_w)[np.newaxis,...]),axis=0)

        # h5utils.save_to_h5(outputLR_filename, "_snr", np.sqrt(snr))
        # h5utils.save_to_h5(outputLR_filename, "_actualSNR", snr_m)
        # h5utils.save_to_h5(outputLR_filename, "_vel_std", vel_std)
        
        # if idx == 0:
        #     # only save once
        #     h5utils.save_to_h5(outputLR_filename, "dx", dx*downsample)
        #     h5utils.save_to_h5(outputLR_filename, "_template_idx", template_idx) # magnitude template idx

        #     h5utils.save_to_h5(outputLR_filename, 'mask', mask_image)
        
        # save vtk
        # grid = pv.ImageData()
        # grid.dimensions = [x_ for x_ in lr_u.shape]
        # grid['mag_u'] = np.reshape( mag_u.astype('double'),-1,order='F')
        # grid['mag_v'] = np.reshape( mag_v.astype('double'),-1,order='F')
        # grid['mag_w'] = np.reshape( mag_w.astype('double'),-1,order='F')
        # grid['u'] = np.reshape( lr_u.astype('double'),-1,order='F')
        # grid['v'] = np.reshape( lr_v.astype('double'),-1,order='F')
        # grid['w'] = np.reshape( lr_w.astype('double'),-1,order='F')
        # grid['mask'] = np.reshape( mask_image.astype('double'),-1,order='F')        
        # grid.save(os.path.join(base_path, f'{new_case_name}_t{idx}.vtk'))
        # print(f"Time taken {(time.time() - start_time):.1f} secs.")
        # # break

        # save LR with noise data
        outputLR_filename = f'{output_dir}/{new_case_name}_t{idx}.h5'
        if os.path.exists(outputLR_filename):
            os.remove(outputLR_filename)
        # Save the downsampled image
        h5utils.save_to_h5(outputLR_filename, "u", np.expand_dims(lr_u,axis=0))
        h5utils.save_to_h5(outputLR_filename, "v", np.expand_dims(lr_v,axis=0))
        h5utils.save_to_h5(outputLR_filename, "w", np.expand_dims(lr_w,axis=0))
        h5utils.save_to_h5(outputLR_filename, "mag_u", np.expand_dims(mag_u,axis=0))
        h5utils.save_to_h5(outputLR_filename, "mag_v", np.expand_dims(mag_v,axis=0))
        h5utils.save_to_h5(outputLR_filename, "mag_w", np.expand_dims(mag_w,axis=0))
        h5utils.save_to_h5(outputLR_filename, "venc_u", np.expand_dims(venc,axis=0))
        h5utils.save_to_h5(outputLR_filename, "venc_v", np.expand_dims(venc,axis=0))
        h5utils.save_to_h5(outputLR_filename, "venc_w", np.expand_dims(venc,axis=0))
        
        # save HR with noise data
        outputLR_filename = f'{output_dir_noise}/{case_name_noise}_t{idx}.h5'
        if os.path.exists(outputLR_filename):
            os.remove(outputLR_filename)
        # Save the downsampled image
        h5utils.save_to_h5(outputLR_filename, "u", np.expand_dims(hr_u_noise,axis=0))
        h5utils.save_to_h5(outputLR_filename, "v", np.expand_dims(hr_v_noise,axis=0))
        h5utils.save_to_h5(outputLR_filename, "w", np.expand_dims(hr_w_noise,axis=0))
        h5utils.save_to_h5(outputLR_filename, "mag_u", np.expand_dims(hr_mag_u,axis=0))
        h5utils.save_to_h5(outputLR_filename, "mag_v", np.expand_dims(hr_mag_v,axis=0))
        h5utils.save_to_h5(outputLR_filename, "mag_w", np.expand_dims(hr_mag_w,axis=0))
        h5utils.save_to_h5(outputLR_filename, "venc_u", np.expand_dims(venc,axis=0))
        h5utils.save_to_h5(outputLR_filename, "venc_v", np.expand_dims(venc,axis=0))
        h5utils.save_to_h5(outputLR_filename, "venc_w", np.expand_dims(venc,axis=0))

        # save noise free LR data
        outputLR_filename = f'{output_dir_nf}/{new_case_name_nf}_t{idx}.h5'
        if os.path.exists(outputLR_filename):
            os.remove(outputLR_filename)
        # Save the downsampled image
        h5utils.save_to_h5(outputLR_filename, "u", np.expand_dims(lr_u_,axis=0))
        h5utils.save_to_h5(outputLR_filename, "v", np.expand_dims(lr_v_,axis=0))
        h5utils.save_to_h5(outputLR_filename, "w", np.expand_dims(lr_w_,axis=0))
        h5utils.save_to_h5(outputLR_filename, "mag_u", np.expand_dims(mag_u_,axis=0))
        h5utils.save_to_h5(outputLR_filename, "mag_v", np.expand_dims(mag_v_,axis=0))
        h5utils.save_to_h5(outputLR_filename, "mag_w", np.expand_dims(mag_w_,axis=0))
        h5utils.save_to_h5(outputLR_filename, "venc_u", np.expand_dims(venc,axis=0))
        h5utils.save_to_h5(outputLR_filename, "venc_v", np.expand_dims(venc,axis=0))
        h5utils.save_to_h5(outputLR_filename, "venc_w", np.expand_dims(venc,axis=0))
        
        print(f"Done! \nSaved in {outputLR_filename}")
        
    snr_u_mean = np.mean(snr_u_list)
    print("\n\nMean SNR_u averaged over all frames\n", round(snr_u_mean,1))
    
    # # TODO : fields for h5 file have to have shape (bins, x, y, z).shape == 4
    # outputLR_filename = f'{output_dir}/{new_case_name}.h5'
    # if os.path.exists(outputLR_filename):
    #     os.remove(outputLR_filename)
    # # Save the downsampled image
    # h5utils.save_to_h5(outputLR_filename, "u", u)
    # h5utils.save_to_h5(outputLR_filename, "v", v)
    # h5utils.save_to_h5(outputLR_filename, "w", w)

    # h5utils.save_to_h5(outputLR_filename, "mag_u", np.expand_dims(mag_u,axis=0))
    # h5utils.save_to_h5(outputLR_filename, "mag_v", np.expand_dims(mag_v,axis=0))
    # h5utils.save_to_h5(outputLR_filename, "mag_w", np.expand_dims(mag_w,axis=0))

    # h5utils.save_to_h5(outputLR_filename, "venc_u", np.expand_dims(venc,axis=0))
    # h5utils.save_to_h5(outputLR_filename, "venc_v", np.expand_dims(venc,axis=0))
    # h5utils.save_to_h5(outputLR_filename, "venc_w", np.expand_dims(venc,axis=0))

    # print(f"Done! \nSaved in {outputLR_filename}")