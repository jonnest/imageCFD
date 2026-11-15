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

# box 1 - right sigmoid
box1_coord = [125,140,12] # [125,140,12] , [125,140,12]
box1_dims = [25,25,25]    # [25,25,25], [30,36,30]
# box 2 - left sigmoid
box2_coord = [125,70,10]
box2_dims = [25,25,25]

# box 1
mask_box1 = np.zeros(mask.shape, dtype=bool)
mask_box1[box1_coord[0]:box1_coord[0]+box1_dims[0],
          box1_coord[1]:box1_coord[1]+box1_dims[1],
          box1_coord[2]:box1_coord[2]+box1_dims[2]] = mask[box1_coord[0]:box1_coord[0]+box1_dims[0],
                                                           box1_coord[1]:box1_coord[1]+box1_dims[1],
                                                           box1_coord[2]:box1_coord[2]+box1_dims[2]]

mask_box2 = np.zeros(mask.shape, dtype=bool)
mask_box2[box2_coord[0]:box2_coord[0]+box2_dims[0],
          box2_coord[1]:box2_coord[1]+box2_dims[1],
          box2_coord[2]:box2_coord[2]+box2_dims[2]] = mask[box2_coord[0]:box2_coord[0]+box2_dims[0],
                                                           box2_coord[1]:box2_coord[1]+box2_dims[1],
                                                           box2_coord[2]:box2_coord[2]+box2_dims[2]]
        
vor0_means = [] # whole mask
vor1_means = [] # box1
vor2_means = [] # box2
for t in range(u_field.shape[-1]):
    vor_mag, _ = compute_vorticity(u_field[...,t], image_res)

    # mask field with box
    vor0_mag = vor_mag[mask.astype(bool)]
    vor1_mag = vor_mag[mask_box1]
    vor2_mag = vor_mag[mask_box2]

    vor0_mean = np.mean(vor0_mag)
    vor1_mean = np.mean(vor1_mag)
    vor2_mean = np.mean(vor2_mag)

    vor0_means += [vor0_mean]    
    vor1_means += [vor1_mean]
    vor2_means += [vor2_mean]

plt.plot(vor0_means, label='whole mask')    
plt.plot(vor1_means, label='right sigmoid')
plt.plot(vor2_means, label='left sigmoid')
plt.xlabel('Caridac cycle')
plt.ylabel('Vorticity [1/s]')
plt.title('Mean vorticity of ROI')
plt.legend()
plt.show()




