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

from flownet_functions import *

def prepare_magnitude(template, vessel, template_mask, case_mask):
    # Get the mean values of template vessel
    vessel_1d = vessel[vessel > 0]
    meanVal = np.mean(vessel_1d)
    stdVal = np.std(vessel_1d)


    # Scale and repeat the template to fit the CFD size
    new_template = scale_and_repeat(template, case_mask)
    vessel = scale_and_repeat(vessel, case_mask)
    template_mask = scale_and_repeat(template_mask, case_mask)

    # Fill in the no signal region with the mean value of the vessel
    nosig_mask = template_mask < 1
    nosig = meanVal * nosig_mask

    # New magnitude image, with nosignal region assigned with mean value
    noisy_vessel = vessel + nosig

    # Cut out the CFD mask from the template
    new_template[case_mask > 0] = 0
    # Fill in the new vessel to the CFD mask
    new_vessel = case_mask * noisy_vessel
    
    # Add them up
    new_magnitude = new_template + new_vessel

    return new_magnitude

if __name__ == '__main__':
    # NOTE: input is velocity image !!!
    # Update your path here
    folder = 'geom3'
   
    base_path = f'/Users/jonathan/Documents/applications/superres/{folder}/time_data/' 

    template_filepath = base_path + 'mag_template_aligned.h5'   
    
    # Change your case name, template idx (see mag_template_aligned.h5), and target SNR here
    # case_name = 'healthy-05mm3_sliced'
    case_name = 'data_HR_r4'

    resolution = 0.4e-3    
    template_idx = 1
    targetSNR = 3.8 # 4.4 ,3.8, 3.3, 2.4
        
    new_case_name = case_name.replace('_HR_', '_LR_')
    
    # make output dir
    output_dir = os.path.join(base_path, f'lr_data_snr{targetSNR}')
    os.makedirs(output_dir, exist_ok=True)
    # -----------------------
    targetSNR = targetSNR**2 # convert to variance

    input_filepath  = f'{base_path}/{case_name}.h5'

    downsample = 2
    crop_ratio = 1 / downsample
    #-----------------------
    is_mask_saved = False 
    
    vnr_list = []
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
    mag_image = prepare_magnitude(template, vessel, template_mask, case_mask)
    
    pad_x, pad_y, pad_z = calculate_pad(mag_image.shape, downsample)

    # Create the magnitude based on the possible values
    ## This is a part of augmentation to make sure we have varying magnitude    
    mag_image = pad(mag_image, pad_x, pad_y, pad_z)
   
    start_time = time.time()
    # number of time frames (frames, x, y, z)
    for idx in range(data_count):
        
        print(f"\nProcessing {idx+1}/{data_count} - SNR {targetSNR}")
        
        # Load the velocity U V W from H5
        with h5py.File(input_filepath, mode = 'r' ) as hf:
            # print(np.asarray(hf['u']))
            hr_u = np.asarray(hf['u'][idx])
            hr_v = np.asarray(hf['v'][idx])
            hr_w = np.asarray(hf['w'][idx])

            # print("hr_u  ", hr_u.shape)
            # max_u = np.asarray(hf['max_u'][idx])
            # max_v = np.asarray(hf['max_v'][idx])
            # max_w = np.asarray(hf['max_w'][idx])

            venc_values =  np.asarray([1., 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]) # in m/s
            # check the max velocity per component (add 10% ceiling to avoid aliasing)
            # max_vel = np.max((max_u, max_v, max_w)) * 1.1
            # randindx = np.random.randint(2) # pick the first or second lowest venc above max vel
            # venc = venc_values[np.where(venc_values > max_vel)][randindx]
            venc = 1.0
            
            # print('venc', venc)
            # print('original shape', hr_u.shape)

            # TODO : if error, comment padding
            hr_u = pad(hr_u, pad_x, pad_y, pad_z)
            hr_v = pad(hr_v, pad_x, pad_y, pad_z) 
            hr_w = pad(hr_w, pad_x, pad_y, pad_z)
            
        # DO the downsampling
        lr_u, mag_u = fft.downsample_phase_img(hr_u, mag_image, venc, crop_ratio, targetSNR)
        lr_v, mag_v = fft.downsample_phase_img(hr_v, mag_image, venc, crop_ratio, targetSNR)
        lr_w, mag_w = fft.downsample_phase_img(hr_w, mag_image, venc, crop_ratio, targetSNR)

        mask_image = pad(case_mask, pad_x, pad_y, pad_z)  ## OWO added
        mask_image = mask_image.astype(np.float32)
        mask_image = ndimage.zoom(mask_image, crop_ratio, order=1) ## OWO added
        mask_image = unpad(mask_image, pad_x//downsample, pad_y//downsample, pad_z//downsample) ## OWO added

        lr_u = unpad(lr_u, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        lr_v = unpad(lr_v, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        lr_w = unpad(lr_w, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        mag_u = unpad(mag_u, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        mag_v = unpad(mag_v, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        mag_w = unpad(mag_w, pad_x//downsample, pad_y//downsample, pad_z//downsample)
       
        ### verify the SNR       
        # mask image
        z_slice = 28
        binary_mask = np.zeros(mask_image.shape,dtype=bool)
        binary_mask[mask_image>0] = True
        # invert binary mask - no fluid 
        nofluid_mask = ~binary_mask
        
        # magnitude of fluid
        umag = np.sqrt(lr_u**2+lr_v**2+lr_w**2)
        
        umag_shape_masked = np.where(binary_mask, umag, 0)
        umag_shape_masked_nonzero = umag_shape_masked[umag_shape_masked != 0]
        # mean magnitude of complete vessel / std of background
        vnr = np.mean(np.abs(umag_shape_masked_nonzero))/np.std(np.abs(umag[nofluid_mask]))
        # vnr = np.mean(np.abs(umag_shape_masked_nonzero))/np.std(np.abs(umag_shape_masked_nonzero))

        print("\n*** vnr   ", vnr, "\n")      
        vnr_list += [vnr]
        # plt.imshow(umag_shape_masked[...,z_slice], cmap='viridis', interpolation='none')
        # plt.colorbar(label='Velocity [m/s]')
        # plt.show()

        # # original calculation
        # mean =  np.mean(img_slice)
        # std = np.std(img_slice)
        # vel_std = std/venc * 100
        # snr_m = (np.sqrt(2) / np.pi) * (venc / std)
        # print(f'Tissue vel_std {vel_std:.1f}% snr_m {snr_m:.1f} targetSNR  {targetSNR:.1f} {np.sqrt(targetSNR):.1f}')

        
        if idx == 0:
            u = np.squeeze(lr_u)[np.newaxis,...]
            v = np.squeeze(lr_v)[np.newaxis,...]
            w = np.squeeze(lr_w)[np.newaxis,...]

        else:
            # For subsequent files, concatenate the new data volumes along the last axis.
            u = np.concatenate((u,np.squeeze(lr_u)[np.newaxis,...]),axis=0)
            v = np.concatenate((v,np.squeeze(lr_v)[np.newaxis,...]),axis=0)
            w = np.concatenate((w,np.squeeze(lr_w)[np.newaxis,...]),axis=0)

        # h5utils.save_to_h5(outputLR_filename, "_targetSNR", np.sqrt(targetSNR))
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

        print(f"Done! \nSaved in {outputLR_filename}")
        
    vnr_frames = np.mean(vnr_list)
    print("\nMean VNR averaged over all frames   ", vnr_frames)
    
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