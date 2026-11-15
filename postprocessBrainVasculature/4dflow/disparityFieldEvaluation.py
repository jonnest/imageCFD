import os,sys
import h5py
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import pyacvd
from surfaceMesh import *
import skimage
from skimage.filters import threshold_multiotsu
from scipy.ndimage import median_filter
from scipy import ndimage
import time
import vtk
import pymeshfix
import nibabel as nib

from fileOperators import *
from flow_field_metrics import compute_vorticity


w_dir = '/Users/jonathan/Documents/eth/02_conferences/ismrm_2025/vein_ismrm/vol000/'

image_res = 0.8e-3

scalars = 'data' # 'scalars' 


# Attempt to load data in either real-imaginary format or magnitude-phase format
loader = loadData()
image = loader.loadFullFlowField(w_dir + 'vol000.h5')

# read segmentation mask
mask_pv = pv.read(w_dir + '/mask.vtk')
mask = np.reshape(mask_pv.point_data[scalars], mask_pv.dimensions, order='F')

# drop reference scan without velocity encoding
image = np.array(image)     
print(image.shape)

if image.shape[3]==4:
    image = image[:, :, :, 1:, :]

u_field = np.angle(image)

# for t in range(u_field.shape[-1]):
#     target_file = w_dir + f'/vein_field_mask_t{t}.h5'
#     if os.path.exists(target_file):
#         os.remove(target_file)
#     file = h5py.File(target_file,'a')
#     file.create_dataset('mask', data = mask[np.newaxis,...], dtype='half')
#     file.create_dataset('u',    data = u_field[np.newaxis,...,0,t], dtype='double')
#     file.create_dataset('v',    data = u_field[np.newaxis,...,1,t], dtype='double')
#     file.create_dataset('w',    data = u_field[np.newaxis,...,2,t], dtype='double')
#     print(u_field[np.newaxis,...,0,t].shape)
#     print(u_field[...,1,t].shape)
#     print(u_field[...,2,t].shape)

#     file.close()

# grid = pv.ImageData()
# grid.dimensions = u_field.shape[:-2]
# grid['u'] = np.reshape(np.where(mask, u_field[...,:,0], 0).astype('double'),-1,order='F')
# grid['v'] = np.reshape(np.where(mask, u_field[...,:,1], 0).astype('double'),-1,order='F')
# grid['w'] = np.reshape(np.where(mask, u_field[...,:,2], 0).astype('double'),-1,order='F')
# grid['mask'] = np.reshape(mask.astype('double'),-1,order='F')
# grid.save(os.path.join(w_dir, f'vein_field_mask_t0.vtk'))


# ------------------------------------------------------------------
# Compute disparty curve for flow field - dot product
# ------------------------------------------------------------------

    
u_field_masked = u_field[mask.astype(bool),...]
u_field_masked_all = u_field_masked.transpose(0, 2, 1).reshape(-1, 3)
# Parameters
subset_size = 200000  # Define the subset size for A and B

# Step 2: Randomly select unique indices for A and B
number_points = u_field_masked_all.shape[0]
indices = np.arange(number_points)
np.random.shuffle(indices)
subset1, subset2 = np.split(indices[:2 * subset_size], 2)

A = u_field_masked_all[subset1, :]
B = u_field_masked_all[subset2, :]

# Initialize an array to store the dot products
dot_products = np.zeros(subset_size)

# Loop over each point in A and B to compute the dot product
for i in range(subset_size):
    a_i = A[i, :]
    b_i = B[i, :]
    a_i_norm = np.linalg.norm(a_i)
    b_i_norm = np.linalg.norm(b_i)
    dot_products[i] = np.dot(a_i, b_i) / (a_i_norm * b_i_norm)  # Dot product of each corresponding vector


# # Step 3: Compute angular disparity between vectors in A and B
# dot_products = np.einsum('ij,ij->j', A, B)  # Dot product of corresponding vectors
# magnitudes_A = np.linalg.norm(A, axis=1)
# magnitudes_B = np.linalg.norm(B, axis=1)
# cos_theta = dot_products / (magnitudes_A * magnitudes_B)
angular_disparities = np.arccos(np.clip(dot_products, -1, 1)) * (180 / np.pi)  # Convert to degrees

# Generate histogram data
counts, bin_edges = np.histogram(angular_disparities, bins=50, density=True)
# Calculate bin centers for plotting the density curve
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Step 4: Plot the angular disparity distribution
plt.figure(figsize=(8, 5))
# plt.hist(angular_disparities, bins=50, density=True, alpha=0.3, color='b', lw=1)
plt.plot(bin_centers, counts, lw=1)
plt.title("Signature Disparity Distribution")
plt.xlabel("Angular Disparity (degrees)")
plt.ylabel("Probability Density")
plt.grid(True)

plt.show()

# ------------------------------------------------------------------
# Compute specific kinetic energy (SKE) - without mass
# ------------------------------------------------------------------
ske_maxs = []
ske_means = []

for t in range(u_field_masked.shape[-1]):
    # calculate specific kinetic energy ske=1/2*u_mag^2
    ske = 0.5 * np.linalg.norm(u_field_masked[...,t], axis=1)**2
    
    ske_means.append(np.mean(ske))
    ske_maxs.append(np.max(ske))
    
plt.plot(ske_means)
plt.title("Mean Kinetic Energy")
plt.xlabel('Caridac cycle')
plt.ylabel(r'Specific kinetic energy $\frac{m^2}{s^2}$')
plt.show()

plt.plot(ske_maxs)
plt.title("Max Kinetic Energy")
plt.xlabel('Caridac cycle')
plt.ylabel(r'Specific kinetic energy $\frac{m^2}{s^2}$')
plt.show()


# ------------------------------------------------------------------
# Spatial cross correlation
# ------------------------------------------------------------------

def spatial_cross_correlation(velocity_field1, velocity_field2):
    """
    Compute spatial cross-correlation at each time frame for two velocity fields.
    Args:
        velocity_field1, velocity_field2: 4D arrays of shape (x, y, z, 3).
    Returns:
        spatial_corr: 1D array of cross-correlation values for each time frame.
    """

    # Flatten the 3D spatial field to a 1D array
    v1_flat = velocity_field1[t].ravel()
    v2_flat = velocity_field2[t].ravel()
    
    # Compute normalized cross-correlation
    correlation = np.corrcoef(v1_flat, v2_flat)[0, 1]
    
    return correlation


corr = []
for i in range(u_field_masked.shape[-1]):
    noise = np.random.normal(0,0.1, u_field_masked[...,t].shape)
    
    corr_ = spatial_cross_correlation(u_field_masked[...,t], u_field_masked[...,t]+noise)
    corr.append(np.mean(corr_))
    
plt.plot(corr)
plt.title("Cross correlation between fields")
plt.show()