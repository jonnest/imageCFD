import h5py
import pyvista as pv
import numpy as np 
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt 
import sys,os,glob,re
from IPython.display import Image as ipImage
from PIL import Image
import imageio
from scipy.ndimage import zoom
from skimage.transform import resize

from gridOperators import *
from filters import *
from fileOperators import *
from functionsRST import *


# Loop through files matching a pattern (eg. h5 files containing "data_LR_")
# fetched by the fetchFiles function. `volume_index` is the index of each file,
# and `target` is the file path.
folder = 'oliver_case'
base_path = f'/Users/jonathan/Documents/applications/superres/{folder}/validation/'


for volume_index,target in enumerate(fetchFiles(base_path + '*',['_LR_'])):
    
    print("volume_index  ", volume_index)
    # Open the current h5 file in read mode.
    data = h5py.File(target,'r')
    print("file ", target.split('/')[-1] )
    #print shape of data


    # If this is the first file, initialize the data arrays by loading and squeezing
    # single data volumes from the file. `np.newaxis` is used to add a new axis,
    # preparing for concatenation along this axis for subsequent files.
    if volume_index == 0:
        u = np.squeeze(data['u'][...])[...,np.newaxis]
        v = np.squeeze(data['v'][...])[...,np.newaxis]
        w = np.squeeze(data['w'][...])[...,np.newaxis]
        mu = np.squeeze(data['mag_u'][...])[...,np.newaxis]
        mv = np.squeeze(data['mag_v'][...])[...,np.newaxis]
        mw = np.squeeze(data['mag_w'][...])[...,np.newaxis]
        # linearly interpolated data
        ul = interpolate_3d_grid(np.squeeze(data['u'][...]), mult_factor=2, method='linear',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis]
        vl = interpolate_3d_grid(np.squeeze(data['v'][...]), mult_factor=2, method='linear',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis]
        wl = interpolate_3d_grid(np.squeeze(data['w'][...]), mult_factor=2, method='linear',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis]

    else:
        # For subsequent files, concatenate the new data volumes along the last axis.
        u = np.concatenate((u,np.squeeze(data['u'][...])[...,np.newaxis]),axis=-1)
        v = np.concatenate((v,np.squeeze(data['v'][...])[...,np.newaxis]),axis=-1)
        w = np.concatenate((w,np.squeeze(data['w'][...])[...,np.newaxis]),axis=-1)
        mu = np.concatenate((mu,np.squeeze(data['mag_u'][...])[...,np.newaxis]),axis=-1)
        mv = np.concatenate((mv,np.squeeze(data['mag_v'][...])[...,np.newaxis]),axis=-1)
        mw = np.concatenate((mw,np.squeeze(data['mag_w'][...])[...,np.newaxis]),axis=-1)  
        # linearly interpolated data
        ul = np.concatenate(interpolate_3d_grid(np.squeeze(data['u'][...]), mult_factor=2, method='linear',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis])
        vl = np.concatenate(interpolate_3d_grid(np.squeeze(data['v'][...]), mult_factor=2, method='linear',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis])
        wl = np.concatenate(interpolate_3d_grid(np.squeeze(data['w'][...]), mult_factor=2, method='linear',axes=(0,1,2),kwindow='hybrid',alpha=0.99)[...,np.newaxis])

    # Store velocity encoding values in a list for further use.
    venc = [data['venc_u'][...],data['venc_v'][...],data['venc_w'][...]]
    print("venc ", venc)
    
    # Close the current h5 file to free resources.
    data.close()   


for volume_index,target in enumerate(fetchFiles(base_path + '*',['_upsampled_'])):
    data = h5py.File(target,'r')
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
        
    # Close the current h5 file to free resources.
    data.close()  


for volume_index,target in enumerate(fetchFiles(base_path + '*',['data_HR_'])):
    data = h5py.File(target,'r')
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
        
    # Close the current h5 file to free resources.
    data.close()   
    
    
print("\nU shape                ", us.shape)
print("U ground truth shape   ", ugt.shape)
print("\nMask shape   ", mask.shape)
print("venc shape   ", venc)

print("\nmax u ", np.max(us))
print("max v ", np.max(vs))
print("max w ", np.max(ws))

print("\nmax u_gt ", np.max(ugt))
print("max v_gt ", np.max(vgt))


# add a new axis to the data
ul = ul[np.newaxis,...]
vl = vl[np.newaxis,...]
wl = wl[np.newaxis,...]

us = us[np.newaxis,...]
vs = vs[np.newaxis,...]
ws = ws[np.newaxis,...]

ugt = ugt[np.newaxis,...]
vgt = vgt[np.newaxis,...]
wgt = wgt[np.newaxis,...]

# target_lr = np.concatenate((ul,vl,wl),axis=0)
target = np.concatenate((us,vs,ws),axis=0)
target_gt = np.concatenate((ugt,vgt,wgt),axis=0)


print("\nBefore resize target shape:   ", target.shape)
print("Target ground truth shape:    ", target_gt.shape)

# resize like this but for every frame in the dataset separately
resized_target = np.zeros(target_gt.shape)
# resized_target_lr = np.zeros(target_gt.shape)

for i in range(target.shape[4]):
    resized_target[...,i]    = resize(target[...,i], target_gt[...,i].shape, order=3, mode='constant', anti_aliasing=True)

# for i in range(target_lr.shape[4]):   
#     resized_target_lr[...,i] = resize(target_lr[...,i], target_lr[...,i].shape, order=3, mode='constant', anti_aliasing=True)

print("\nResized target shape:         ", resized_target.shape)
print( "Target ground truth shape:    ", target_gt.shape)
print( "Resized target LR shape:      ", target_lr.shape)

binary_mask = np.zeros(mask.shape,dtype=bool)
binary_mask[mask>0.5] = True
binary_mask_non_fluid = np.logical_not(binary_mask)
print("\nBinary mask shape:            ", binary_mask.shape)

resized_target_v = np.sqrt(np.sum(resized_target**2,axis=0))
target_gt_v = np.sqrt(np.sum(target_gt**2,axis=0))
masked_target = resized_target[:,binary_mask]
nf_masked_target = resized_target[:,binary_mask_non_fluid]
masked_target_gt = target_gt[:,binary_mask]
nf_masked_target_gt = target_gt[:,binary_mask_non_fluid]

# apply binary max to each frame of the array resized_target (214,66,46,20) the frame number is the last dimension
masked_velocity = []
masked_velocity_gt = []
masked_target_list = []
masked_target_gt_list = []
for i in range(binary_mask.shape[3]):
    masked_velocity.append(resized_target_v[binary_mask[...,i],i])
    masked_velocity_gt.append(target_gt_v[binary_mask[...,i],i])
    masked_target_list.append(resized_target[:,binary_mask[...,i],i])
    masked_target_gt_list.append(target_gt[:,binary_mask[...,i],i])



# compute the average velocity magnitude for each frame
average_velocity = []
average_velocity_gt = []
for i in range(len(masked_velocity)):
    average_velocity.append(np.mean(masked_velocity[i]))
    average_velocity_gt.append(np.mean(masked_velocity_gt[i]))


# plot the average velocity over time
plt.plot(average_velocity, label='Upsampled velocity', marker='o')
plt.plot(average_velocity_gt, label='Ground truth velocity', marker='o')
# make the x label unitary range from 0 to 19
plt.xticks(np.arange(-2, 22, 2))
plt.xlabel('Frame')
plt.ylabel('Average velocity magnitude (m/s)')
plt.legend()
plt.title('Average velocity magnitude over time')
plt.grid()
plt.show()