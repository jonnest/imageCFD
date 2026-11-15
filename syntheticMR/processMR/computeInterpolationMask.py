import os
import pyvista as pv 
import h5py
import numpy as np 
from surfaceMesh import *
from operationsH5 import sequentialLoad



### data folder
target = '/Users/jonathan/Documents/applications/cfd/eccomas_case/CFD/'

vessel = 'v1'
file_name = f'cfd_interpolation_grid_{vessel}.vtu'
mr_mask_name = f'mr_interpolation_mask_{vessel}.vtk'


### mr data folder
w_dir = '/Users/jonathan/Documents/applications/cfd/eccomas_case/syntheticMR/'

case_id = 'embedded_0.8mm_50ms_SNR15'
mr_folder = os.path.join(w_dir,case_id)

mr_files_all = os.listdir(mr_folder)
mr_files = []
for file in mr_files_all:
    # Check if exclude_string is not in the file name
    if 'embedded' in file:
        mr_files.append(file)
# pick the first
mr_file = mr_files[0]

r_iso = 0.8 * 1e-3


surface_file = target + '/' + file_name
surface = pv.read(surface_file)




grid_voxel_size = r_iso
save=True

########################################################################################
for mr_file in mr_files:
    # get velocity field
    with h5py.File(os.path.join(mr_folder,mr_file), 'r') as f: 
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

########################################################################################

pyskeleton, origin, spacing, voxel_mask = pyskeletonize(surface,grid_voxel_size=grid_voxel_size,
                                                        voxelization_size=grid_voxel_size,
                                                        estimate_distance=True,coarse_voxel_size=grid_voxel_size*3,
                                                        distance_tolerance=1e-3, save=save, target=target)


# here use voxelized geometry of pyskeleton algorithm
# get zero padding for voxelized geometry to fill up similarly to flow dimensions
# flow_im.dimensions = (vel_mag.shape[0]+1,
#                       vel_mag.shape[1]+1,
#                       vel_mag.shape[2]+1)
# padding = [(abs(int((flow_im.bounds[2*i]  - voxel_mask.bounds[2*i]  )/spacing[i])),
#             abs(int((flow_im.bounds[2*i+1]- voxel_mask.bounds[2*i+1])/spacing[i])) ) for i in range(3)]

padding = [(abs(int((0                       - voxel_mask.bounds[2*i]  )/spacing[i])),
            abs(int((data.shape[i]+1/spacing - voxel_mask.bounds[2*i+1])/spacing[i])) ) for i in range(3)]

# Sample points (segmentation mask of voxelized geometry) to new shape (original velocity shape mask)   
to_sample = np.reshape(voxel_mask['vtkValidPointMask'], voxel_mask.dimensions, order='F')
padded_sample = np.pad(to_sample, pad_width=padding) 
    
padded_voxel = pv.ImageData()
padded_voxel.dimensions = (padded_sample.shape[0]+1,
                            padded_sample.shape[1]+1,
                            padded_sample.shape[2]+1)

padded_voxel.cell_data[f'data'] = np.reshape(padded_sample,-1,order='F')
padded_voxel.spacing = spacing  
padded_voxel.save(target + '/' + mr_mask_name)