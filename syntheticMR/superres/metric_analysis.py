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
from tqdm import tqdm


def fetchFiles(path,keyword_in=None,keyword_out=None,sort=True):
    """
    Fetches files from a given path, optionally filtering them by inclusion or exclusion keywords,
    and sorts them in natural order if specified.
    
    Parameters:
    - path: The directory path where to search for files.
    - keyword_in: Optional list of keywords to include files. Only files containing any of these keywords will be included.
    - keyword_out: Optional list of keywords to exclude files. Any file containing any of these keywords will be excluded.
    - sort: Boolean indicating whether the returned file list should be sorted in natural order. Defaults to True.
    
    Returns:
    - List of file paths, filtered and sorted according to the specified criteria.
    """

    # Helper function to convert text to integers if possible,
    # which helps in sorting files naturally.
    def atoi(text):
        return int(text) if text.isdigit() else text

    # Split file names into parts that are integers or non-integers,
    # and convert integers to actual integers for natural sorting.
    def natural_keys(text):
        return [ atoi(c) for c in re.split('(\d+)',text) ]

    # Use glob to find all files matching the path pattern.
    files=glob.glob(path)

    # If inclusion keywords are specified, filter the files to include
    # only those containing any of the inclusion keywords.
    if keyword_in:
        keywordFilter=keyword_in
        files = [sent for sent in files
            if any(word in sent for word in keywordFilter)]

    # If exclusion keywords are specified, filter the files to exclude
    # those containing any of the exclusion keywords.
    if keyword_out:
        keywordFilter=keyword_out
        files = [sent for sent in files
            if not any(word in sent for word in keywordFilter)]

    # If sorting is enabled, sort the files using the natural keys method,
    # which considers numerical values for sorting parts of the file names.
    if sort:
        files.sort(key=natural_keys)
        return files
    else:
        return files
    
    
# Loop through files matching a pattern (eg. h5 files containing "data_LR_")
# fetched by the fetchFiles function. `volume_index` is the index of each file,
# and `target` is the file path.


# w_dir = '/Users/jonathan/Documents/applications/superres/data_local/'
base_path = '/Users/jonathan/Documents/applications/superres/output_analysis/'

num_frames = 0
for volume_index,target in enumerate(fetchFiles(base_path + '*',['data_LR_'])):
    # Open the current h5 file in read mode.
    data = h5py.File(target,'r')
    
    # Store velocity encoding values in a list for further use.
    venc = [data['venc_u'][...],data['venc_v'][...],data['venc_w'][...]]

    # Close the current h5 file to free resources.
    data.close()   

for volume_index,target in enumerate(fetchFiles(base_path + '*',['data_up_orig_'])):
    data = h5py.File(target,'r')
    if volume_index == 0:
        # Initialize high-resolution data arrays for the first file.
        us_o = np.squeeze(data['u'][...])[...,np.newaxis]
        vs_o = np.squeeze(data['v'][...])[...,np.newaxis]
        ws_o = np.squeeze(data['w'][...])[...,np.newaxis]
    else:
        # Concatenate new data volumes to the existing arrays for subsequent files.
        us_o = np.concatenate((us_o,np.squeeze(data['u'][...])[...,np.newaxis]),axis=-1)
        vs_o = np.concatenate((vs_o,np.squeeze(data['v'][...])[...,np.newaxis]),axis=-1)
        ws_o = np.concatenate((ws_o,np.squeeze(data['w'][...])[...,np.newaxis]),axis=-1)
        
    # Close the current h5 file to free resources.
    data.close()   

for volume_index,target in enumerate(fetchFiles(base_path + '*',['data_no_overfit_'])):
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

for volume_index,target in enumerate(fetchFiles(base_path + '*',['data_zero_filling_'])):
    data = h5py.File(target,'r')
    if volume_index == 0:
        # Initialize high-resolution data arrays for the first file.
        us_zero = np.squeeze(data['u'][...])[...,np.newaxis]
        vs_zero = np.squeeze(data['v'][...])[...,np.newaxis]
        ws_zero = np.squeeze(data['w'][...])[...,np.newaxis]
    else:
        # Concatenate new data volumes to the existing arrays for subsequent files.
        us_zero = np.concatenate((us_zero,np.squeeze(data['u'][...])[...,np.newaxis]),axis=-1)
        vs_zero = np.concatenate((vs_zero,np.squeeze(data['v'][...])[...,np.newaxis]),axis=-1)
        ws_zero = np.concatenate((ws_zero,np.squeeze(data['w'][...])[...,np.newaxis]),axis=-1)
        
    # Close the current h5 file to free resources.
    data.close() 

for volume_index,target in enumerate(fetchFiles(base_path + '*',['data_linear_'])):
    data = h5py.File(target,'r')
    if volume_index == 0:
        # Initialize high-resolution data arrays for the first file.
        us_linear = np.squeeze(data['u'][...])[...,np.newaxis]
        vs_linear = np.squeeze(data['v'][...])[...,np.newaxis]
        ws_linear = np.squeeze(data['w'][...])[...,np.newaxis]
    else:
        # Concatenate new data volumes to the existing arrays for subsequent files.
        us_linear = np.concatenate((us_linear,np.squeeze(data['u'][...])[...,np.newaxis]),axis=-1)
        vs_linear = np.concatenate((vs_linear,np.squeeze(data['v'][...])[...,np.newaxis]),axis=-1)
        ws_linear = np.concatenate((ws_linear,np.squeeze(data['w'][...])[...,np.newaxis]),axis=-1)
        
    # Close the current h5 file to free resources.
    data.close()   

for volume_index,target in enumerate(fetchFiles(base_path + '*',['data_cubic_'])):
    data = h5py.File(target,'r')
    if volume_index == 0:
        # Initialize high-resolution data arrays for the first file.
        us_cubic = np.squeeze(data['u'][...])[...,np.newaxis]
        vs_cubic = np.squeeze(data['v'][...])[...,np.newaxis]
        ws_cubic = np.squeeze(data['w'][...])[...,np.newaxis]
    else:
        # Concatenate new data volumes to the existing arrays for subsequent files.
        us_cubic = np.concatenate((us_cubic,np.squeeze(data['u'][...])[...,np.newaxis]),axis=-1)
        vs_cubic = np.concatenate((vs_cubic,np.squeeze(data['v'][...])[...,np.newaxis]),axis=-1)
        ws_cubic = np.concatenate((ws_cubic,np.squeeze(data['w'][...])[...,np.newaxis]),axis=-1)
        
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






1/0


print("max u ", np.max(us))
print("max v ", np.max(vs))
print("max w ", np.max(ws))

print("max u_gt ", np.max(ugt))
print("max v_gt ", np.max(vgt))
print("max w_gt ", np.max(wgt))


# add a new axis to the data
us = us[np.newaxis,...]
vs = vs[np.newaxis,...]
ws = ws[np.newaxis,...]

ugt = ugt[np.newaxis,...]
vgt = vgt[np.newaxis,...]
wgt = wgt[np.newaxis,...]

us_o = us_o[np.newaxis,...]
vs_o = vs_o[np.newaxis,...]
ws_o = ws_o[np.newaxis,...]

us_zero = us_zero[np.newaxis,...]
vs_zero = vs_zero[np.newaxis,...]
ws_zero = ws_zero[np.newaxis,...]

us_linear = us_linear[np.newaxis,...]
vs_linear = vs_linear[np.newaxis,...]
ws_linear = ws_linear[np.newaxis,...]

us_cubic = us_cubic[np.newaxis,...]
vs_cubic = vs_cubic[np.newaxis,...]
ws_cubic = ws_cubic[np.newaxis,...]



target = np.concatenate((us,vs,ws),axis=0)
target_o = np.concatenate((us_o,vs_o,ws_o),axis=0)
target_zero = np.concatenate((us_zero,vs_zero,ws_zero),axis=0)
target_linear = np.concatenate((us_linear,vs_linear,ws_linear),axis=0)
target_cubic = np.concatenate((us_cubic,vs_cubic,ws_cubic),axis=0)
target_gt = np.concatenate((ugt,vgt,wgt),axis=0)
print("Upsampled shape: ", target.shape)
print("Ground truth shape: ", target_gt.shape)


#visualize a slice of ugt and us with matplotlib
slice_index = 20

# Create a figure with two subplots.
fig,axs = plt.subplots(1,3,figsize=(12,6))

max_v = np.max(target_gt[0,...,slice_index,1])
min_v = np.min(target_gt[0,...,slice_index,1])

# Plot the slice of the ground truth data.
axs[0].imshow(target_gt[0,...,slice_index,1],cmap='viridis', vmin=min_v, vmax=max_v)

# Plot the slice of the upsampled data.
axs[1].imshow(target[0,...,slice_index,1],cmap='viridis', vmin=min_v, vmax=max_v)

axs[2].imshow(target_o[0,...,slice_index,1],cmap='viridis', vmin=min_v, vmax=max_v)



# Set titles for the subplots.
axs[0].set_title('U Ground Truth')
axs[1].set_title('U Trained our data')
axs[2].set_title('U Original data')

#add color bars
fig.colorbar(axs[0].imshow(target_gt[0,...,slice_index,1],cmap='viridis',vmin=min_v, vmax=max_v), ax=axs[0])
fig.colorbar(axs[1].imshow(target[0,...,slice_index,1],cmap='viridis',vmin=min_v, vmax=max_v), ax=axs[1])
fig.colorbar(axs[2].imshow(target_o[0,...,slice_index,1],cmap='viridis',vmin=min_v, vmax=max_v), ax=axs[2])

# Display the figure.
plt.show()



print("Before resize target shape: ", target.shape)
print("Target ground truth shape: ", target_gt.shape)
#resize like this but for every frame in the dataset separately
resized_target = np.zeros(target_gt.shape)
resized_target_o = np.zeros(target_gt.shape)
resized_target_zero = np.zeros(target_gt.shape)
resized_target_linear = np.zeros(target_gt.shape)
resized_target_cubic = np.zeros(target_gt.shape)

for i in tqdm(range(num_frames)):
    resized_target[...,i] = resize(target[...,i], target_gt[...,i].shape, order=1, mode='constant', anti_aliasing=True)
    resized_target_o[...,i] = resize(target_o[...,i], target_gt[...,i].shape, order=1, mode='constant', anti_aliasing=True)
    resized_target_zero[...,i] = resize(target_zero[...,i], target_gt[...,i].shape, order=1, mode='constant', anti_aliasing=True)
    resized_target_linear[...,i] = resize(target_linear[...,i], target_gt[...,i].shape, order=1, mode='constant', anti_aliasing=True)
    resized_target_cubic[...,i] = resize(target_cubic[...,i], target_gt[...,i].shape, order=1, mode='constant', anti_aliasing=True)

print("Resized target shape: ", resized_target.shape, "    Target ground truth shape: ", target_gt.shape)



print("Before resize target shape: ", target.shape)
print("Target ground truth shape: ", target_gt.shape)
#resize like this but for every frame in the dataset separately
resized_target = np.zeros(target_gt.shape)
resized_target_o = np.zeros(target_gt.shape)
resized_target_zero = np.zeros(target_gt.shape)
resized_target_linear = np.zeros(target_gt.shape)
resized_target_cubic = np.zeros(target_gt.shape)

for i in tqdm(range(num_frames)):
    resized_target[...,i] = resize(target[...,i], target_gt[...,i].shape, order=1, mode='constant', anti_aliasing=True)
    resized_target_o[...,i] = resize(target_o[...,i], target_gt[...,i].shape, order=1, mode='constant', anti_aliasing=True)
    resized_target_zero[...,i] = resize(target_zero[...,i], target_gt[...,i].shape, order=1, mode='constant', anti_aliasing=True)
    resized_target_linear[...,i] = resize(target_linear[...,i], target_gt[...,i].shape, order=1, mode='constant', anti_aliasing=True)
    resized_target_cubic[...,i] = resize(target_cubic[...,i], target_gt[...,i].shape, order=1, mode='constant', anti_aliasing=True)

print("Resized target shape: ", resized_target.shape, "    Target ground truth shape: ", target_gt.shape)


def directional_error(vectors_a, vectors_b):
    # Dot product between pairs of vectors
    dot_product = np.abs(np.einsum('ij,ij->j', vectors_a, vectors_b))

    # Norm (magnitude) of each vector
    norm_a = np.linalg.norm(vectors_a, axis=0) + 1e-6
    norm_b = np.linalg.norm(vectors_b, axis=0) + 1e-6
    return np.mean(1-dot_product/norm_a/norm_b)*100
def vectorial_rmse(vectors_a, vectors_b):
    # Vectorial difference
    diff = np.mean(np.linalg.norm(vectors_a-vectors_b,axis=0)**2)
    #Norm (magnitude) of reference target (here: vectors_b)
    norm = np.mean(np.linalg.norm(vectors_b,axis=0)**2) + 1e-6
    
    return np.sqrt(diff/norm)*100
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


### Metric analysis for whole geometry our data

print("Fluid region:\n")
dir_error = directional_error(masked_target.reshape(3,-1),masked_target_gt.reshape(3,-1))
vec_rmse = vectorial_rmse(masked_target.reshape(3,-1),masked_target_gt.reshape(3,-1))
rmse_val = rmse(masked_target,masked_target_gt)
print(f"Directional error: {dir_error:.2f}%")
print(f"Vectorial RMSE: {vec_rmse:.2f}%")
print(f"RMSE: {rmse_val:.3f}")


print("Fluid + non fluid region:\n")
dir_error = directional_error(resized_target.reshape(3,-1),target_gt.reshape(3,-1))
vec_rmse = vectorial_rmse(resized_target.reshape(3,-1),target_gt.reshape(3,-1))
rmse_val = rmse(resized_target,target_gt)
print(f"Directional error: {dir_error:.2f}%")
print(f"Vectorial RMSE: {vec_rmse:.2f}%")
print(f"RMSE: {rmse_val:.3f}")

print("Non fluid region:\n")
dir_error = directional_error(nf_masked_target.reshape(3,-1),nf_masked_target_gt.reshape(3,-1))
vec_rmse = vectorial_rmse(nf_masked_target.reshape(3,-1),nf_masked_target_gt.reshape(3,-1))
rmse_val = rmse(nf_masked_target,nf_masked_target_gt)
print(f"Directional error: {dir_error:.2f}%")
print(f"Vectorial RMSE: {vec_rmse:.2f}%")
print(f"RMSE: {rmse_val:.3f}")