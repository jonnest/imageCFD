""" Embedding the CFD simulation into the MRI background

"""
import h5py
import numpy as np 
import matplotlib.pyplot as plt 
import pyvista as pv 
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage as ndi
import os,sys
import skimage
from scipy.ndimage import zoom as zoom
import sys,os,glob,re
from fileOperators import fetchFiles
from gridOperators import interpolate_3d_grid


target = os.path.abspath('/Users/jonathan/Documents/applications/cfd/eccomas_case/')    
mr_folder = 'synthesizedMR'
cfd_folder = 'CFD'

# time averaging in milli seconds, default: 50, 70, 90
time_avg = 50
nr_bins = 20 # default 20

# isotropioc resolution
r_iso = 0.4 * 1e-3
# total of 100 simulation time steps, for selector of CFD time steps
time_scale = 100/nr_bins

# time averaging kernel
delta = {
    50 : [-2,-1,0,1,2],
    70 : [-3,-2,-1,0,1,2,3],
    90 : [-4,-3,-2,-1,0,1,2,3,4]
}

# make directory 
save_dir = target + f'/embedded_{time_avg}ms'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# loop over velocity encodings
for venc_index in range(3):
    # loop over time bins
    for time_index in range(nr_bins):
        mri = h5py.File(target + f'/{mr_folder}/0.4mm_50ms_SNRinf_venc{venc_index}_t{time_index}.hdf5')
        magnitude_full = mri['3D_recon']['magnitude'][...]
        phase_full = mri['3D_recon']['phase'][...]
        mask_full = np.empty(magnitude_full.shape)
        
        # loop over all CFD geometries, default 4
        for geom_index in range(4):
            
            # simulation time 
            n_time = int(time_index * time_scale)
            delta_average = delta[time_avg]
            # get cfd files : adapt CFD folder names
            flow_files = fetchFiles(target + f'/{cfd_folder}/geom{geom_index+1}/*', ['vessel_'],['wall_'])
            
            U = np.empty(pv.read(flow_files[0])['U'].shape)
            
            # time averaging over the kernel
            for target_avg in delta_average:
                U+=pv.read(flow_files[(n_time+target_avg)%len(flow_files)])['U']/len(delta_average)

            flow = pv.read(flow_files[0])
            # scale flow up to resolution of image space in pixel with r=0.0004
            # after rescaling, CFD flow and MRI are in the same image space
            flow = flow.scale(1/r_iso)

            # Remove all cell data attributes from the mesh except for 'U' (velocity vector) and 'RST' (stress tensor)
            for attribute in flow.cell_data.keys():
                if attribute == 'U' or attribute == 'RST':
                    pass
                else:
                    flow.cell_data.remove(attribute)
            flow['U']=U

            # get the whole bound of the CFD domain
            dxyz = (np.array(flow.bounds)).astype('int')
            
            # extend the image space by 5
            dxyz+=[-5,5,-5,5,-5,5]
            print("dxyz  ", dxyz)
            dxyz[dxyz<0]=0
            print("dxyz  ", dxyz)
            
            # crop the MRI image space down to the extend of the CFD
            magnitude = magnitude_full[dxyz[0]:dxyz[1],dxyz[2]:dxyz[3],dxyz[4]:dxyz[5]]
            phase = phase_full[dxyz[0]:dxyz[1],dxyz[2]:dxyz[3],dxyz[4]:dxyz[5]]
            # get the grid origin
            corner = np.array([dxyz[0],dxyz[2],dxyz[4]])
            crop_size = magnitude.shape

            mult_factor = 2
            venc = 1
            # upsample MR data : interpolate the MR full signal up by factor mult_factor
            image_int = interpolate_3d_grid(magnitude*np.exp(1j*phase), mult_factor, method='linear',axes=(0,1,2))
            
            magnitude = np.abs(image_int)
            phase = np.angle(image_int)
            
            # create and save MRI (without CFD flow) as grid
            grid = pv.ImageData()
            grid.dimensions = [x_ for x_ in magnitude.shape]
            grid.origin = corner
            grid.spacing = (1/mult_factor,1/mult_factor,1/mult_factor)
            grid['magnitude'] = np.reshape(magnitude,-1,order='F').astype('float')
            grid['phase'] = np.reshape(phase,-1,order='F').astype('float')
            # grid.save(target + '/grid_no_cfd.vtk')
            
            # sample the CFD flow data onto the MR grid
            grid = grid.sample(flow)
            grid['phase'] = grid['U'][:,venc_index] * (grid['vtkValidPointMask']>0.5) / venc*np.pi + grid['phase']*(grid['vtkValidPointMask']<0.5)
            
            # get magnitude and phase from grid and downsample again
            image_up = np.reshape(grid['magnitude'],grid.dimensions,order='F')*np.exp(1j*np.reshape(grid['phase'],grid.dimensions,order='F'))
            image_down = interpolate_3d_grid(image_up, 1/mult_factor, method='fft',axes=(0,1,2))
            
            # zoom opertation
            print("image down ", image_down.shape)
            image_down = zoom(image_down, np.array(crop_size)/np.array(image_down.shape))
            print("image down ", image_down.shape)

            mask_up = np.reshape(grid['vtkValidPointMask'],grid.dimensions,order='F')
            # downsampling
            mask_down = interpolate_3d_grid(mask_up, 1/mult_factor, method='fft',axes=(0,1,2))
            # zoom
            mask_down = zoom(np.abs(mask_down),np.array(crop_size)/np.array(mask_down.shape))
            
            # final image with background MRI and CFD in vessels
            magnitude_full[dxyz[0]:dxyz[1],dxyz[2]:dxyz[3],dxyz[4]:dxyz[5]] = np.abs(image_down)
            phase_full[dxyz[0]:dxyz[1],dxyz[2]:dxyz[3],dxyz[4]:dxyz[5]] = np.angle(image_down)
            mask_full[dxyz[0]:dxyz[1],dxyz[2]:dxyz[3],dxyz[4]:dxyz[5]] = mask_down

        file = h5py.File(target + f'/embedded_{time_avg}ms/embedded_0.4mm_50ms_SNRinf_venc{venc_index}_t{time_index}.hdf5','a')
        group = file.create_group('3D_recon')
        group['magnitude'] = magnitude_full.astype('half')
        group['phase'] = phase_full.astype('half')
        if venc_index==0:
            if time_index==0:
                group['mask'] = mask_full.astype('half')
        file.close()
        
        grid = pv.ImageData()
        grid.dimensions = magnitude_full.shape
        grid['u'] = np.reshape(phase_full.astype('double'), -1, order='F')
        grid['mag_u'] = np.reshape(magnitude_full.astype('double'), -1, order='F')
        grid['mask'] = np.reshape(mask_full.astype('double'), -1, order='F')
        grid.save(os.path.join(target + f'/embedded_{time_avg}ms/org_0.4mm_50ms_HR_t{time_index}.vtk'))
        
        
        1/0