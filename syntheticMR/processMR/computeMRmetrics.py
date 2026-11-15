import sys,os,re
import numpy as np
import matplotlib.pyplot as plt
from computeMetrics import * 
from surfaceMesh import *
from gridOperators import getMaskFromSurface
from fileOperators import *


### data folder
w_dir = '/Users/jonathan/Documents/applications/cfd/eccomas_case/syntheticMR/'

# case_id = 'embedded_0.8mm_50ms_SNR15'
case_id = 'embedded_0.4mm_50ms_SNR15'

target = os.path.join(w_dir,case_id)


mr_files_all = os.listdir(target)

# TODO : fetch files
mr_files = []
for file in mr_files_all:
    # Check if exclude_string is not in the file name
    if 'embedded' in file:
        mr_files.append(file)
mr_files.sort()

mr_files = [mr_files[0]]


# synching the phase of mr and cfd
sync_phase = 7
mr_files = mr_files[sync_phase:]+mr_files[:sync_phase]

# spline for centerlines
spline_dir = '/Users/jonathan/Documents/data/deepFlow/segmentations/polar_tinnitus_eccomas/teccomas_v1'

# homogeneous interpolation grid
w_dir_cfd = '/Users/jonathan/Documents/applications/cfd/eccomas_case/CFD/'
interpolation_grid_name = 'cfd_interpolation_grid_v1.vtu'
interpolation_grid_file = os.path.join(w_dir_cfd, interpolation_grid_name)
interpolation_grid = pv.read(interpolation_grid_file)

############################################################################
############################################################################

# nRMSE, SSIM and vectorial rmse of U and angle error of U 
# vectorial error is: abs(Umri - Ucfd)/abs(Ucfd)

############################################################################
def extract_first_float(input_string):
    # Regular expression to find a float number
    float_pattern = r'\d+\.\d+'
    match = re.search(float_pattern, input_string)
    if match:
        return float(match.group())
    else:
        return None


############################################################################
### parameters
# grid resolution in m
r_grid = extract_first_float(case_id) * 1e-3

vmax=1
extra_velocity_margin = 1

# venc = np.array([np.ceil(2 * vmax * extra_velocity_margin) / 2])/3
venc = vmax

############################################################################
### eccomas case
# outflow boundary: left, right branch
# v1 outflow : 1, 2
spline_name = 'spline_2.vtk'
spline_string = spline_name.split('.')[0]
spline_path = os.path.join(spline_dir, spline_name)
   
spline = pv.read(spline_path)
spline_pts = spline.points
spline_grad = np.gradient(spline_pts)[0]

############################################################################
### compute voxel mask of interpolation_grid_surface
save = True
# grid_voxel_size = r_grid / 2
grid_voxel_size = 0.4 * 1e-3

surface_interpolation_grid = interpolation_grid.extract_surface()

_, origin, spacing, voxel_mask = pyskeletonize(surface_interpolation_grid,grid_voxel_size=grid_voxel_size,
                                                        voxelization_size=grid_voxel_size,
                                                        estimate_distance=True,coarse_voxel_size=grid_voxel_size*3,
                                                        distance_tolerance=1e-3, save=save, target=target)

############################################################################
### flow rate 
# print('\nExtracting the boundary flow rate ...')
q = []
for mr_file in mr_files:
    # get velocity field
    with h5py.File(os.path.join(target,mr_file), 'r') as f: 
        try:
            data = f['3D_recon']['real'][:] + 1j*f['3D_recon']['imag'][:]
            data = np.squeeze(data)
            f.close()
            vel_mag = np.sqrt(np.sum(np.square(np.angle(np.squeeze(data))), axis=3))
            # print(f"Loaded {mr_file} data in real-imaginary format")
        except:
            data = f['3D_recon']['phase'][:]
            data = np.squeeze(data)
            f.close()
            vel_mag = np.sqrt(np.sum(np.square(np.squeeze(data)), axis=3))
            # print(f"Loaded {mr_files} data in magnitude-phase format")

    ### zero pad the velocity field in z-direction - since 
    # Generate padded image
    n_padding_voxels = 50
    # image = np.pad(image,((n_padding_voxels,n_padding_voxels),(n_padding_voxels,n_padding_voxels),(n_padding_voxels,n_padding_voxels),(0,0),(0,0)),mode='edge')        image = np.pad(image,((n_padding_voxels,n_padding_voxels),(n_padding_voxels,n_padding_voxels),(n_padding_voxels,n_padding_voxels),(0,0),(0,0)),mode='edge')
    vel_mag = np.pad(vel_mag,((0,0),(0,0),(0,n_padding_voxels)),mode='edge')

    ### extract flowrates at all boundaries
    # mask is numpy nd array
    # print('\nGenerating the mask image ...')
    mask_image = getMaskFromSurface(vel_mag, voxel_mask, r_iso=r_grid, target=target)
    mask = np.reshape(mask_image['data'], vel_mag.shape[:3], order='F')
    masked_flow = mask * vel_mag
    
    masked_flow_image = mask_image.copy()
    masked_flow_image['u_mag'] = np.reshape(masked_flow, -1, order='F')
    masked_flow_image.save(target + '/masked_flow_image.vtk')
    
    # flowrate evaluation at spline index XX
    spline_index = 900
    
    ### compute flow rate directly on masked_flow
    ### resolution of mr scan
    # q += [extractMRFlowRate_NoImage(masked_flow, spline_pts, spline_index, venc, r_grid, target=target)]
    
    ### sampling the voxel mask of the surface of the interpolation grid onto the interpolation grid itself
    sampled_field = interpolation_grid.sample(masked_flow_image)
    sampled_field.save(target + '/sampled_mr_field.vtk')
    
    # q += [extractMRFlowRate(sampled_field, spline_pts, spline_index, venc, r_grid, target=target)]
    
    
# save data
q_dict = {}
q_dict['q'] = q
q_path = os.path.join(target, 'flowrates.txt')

write_dict_to_file(q_dict, q_path)
############################################################################
# plot
q = np.array(q)
### nan values
# q = q[~np.isnan(q)]

## q = np.nan_to_num(q, nan=0.0)


# q_filt = scp.signal.savgol_filter(q,int(len(q)/2), 2)

plt.plot(np.array(q)*1e6, 'o', label='q')
# plt.plot(np.array(q_filt)*1e6, label='q filt')


plt.ylabel(r'flow rate [ml/s]')
plt.xlabel(r'cardiac cycle')
# plt.title(f'flowrate') 
plt.savefig(w_dir + '/' + f'flow_rate_{case_id}_{spline_string}.png', dpi=200)
plt.show()


############################################################################


