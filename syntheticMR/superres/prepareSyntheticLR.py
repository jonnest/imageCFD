import h5py
import pyvista as pv
import numpy as np 
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt 
import sys,os,glob,re
from scipy.ndimage import zoom as zoom
import sys,os,glob,re

from gridOperators import *
from filters import *
from fileOperators import *
from functionsRST import *


venc   = np.array([1.0])
folder = 'geom3'

reduction_factor = 0.5

# Resolution of the image in meters
image_resolution = 0.5e-3 # 1.25e-3  # default 1.25e-3

# Set the velocity-to-noise ratio (vnr) and the reduction factor for downsampling.
vnr = 10 # Ferdian(cerebro) : 3, 6 
print("vnr                 ", vnr)
print("image resolution    ", image_resolution)
# TODO : complex noise does not work
# noise only on velocity data
add_complex_noise = True

# downsampling the HR image
kwindow = 'hybrid' # gaussian, tukey, hybrid, None
method = 'fft'  # fft, linear    

    
base_path =  f'/Users/jonathan/Documents/applications/superres/{folder}/org_data/'
target_path =  f'/Users/jonathan/Documents/applications/superres/{folder}/data_local/'

#create target path if it doesn't exists
if not os.path.exists(target_path):
    os.makedirs(target_path)
    
# Define the MRI encoding directions as unit vectors along the x, y, and z axes.
enc_dirs = np.array([[1., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]])

# Calculate the encoding strength (kv_rec) based on the encoding velocity (venc).
# It starts with 0 (no encoding) and then uses the formula π/venc for each direction.
kv_rec = np.concatenate(([0], np.pi / venc))
kv_rec = kv_rec[:, None]  # Add a new axis to make kv_rec a column vector.
kv_rec = np.tile(kv_rec, (1, len(enc_dirs)))  # Repeat kv_rec for each encoding direction.

# Iterate over volume files matching a specific pattern, such as 'volume_'
for volume_index,target in enumerate(fetchFiles(base_path + '*',['volume_'])):
    print("\nVolume index:       ", volume_index)
    print("Target              ", target.split('/')[-1])
    # Read the mesh data from the current target file
    mesh = pv.read(target)
    
    # Remove all cell data attributes from the mesh except for 'U' (velocity vector) and 'RST' (stress tensor)
    for attribute in mesh.cell_data.keys():
        if attribute == 'U' or attribute == 'RST':
            pass
        else:
            mesh.cell_data.remove(attribute)
            
    # Initialize image parameters only for the first volume
    if volume_index == 0:
        image_padding = 0e-3  # 10e-3 # Padding around the image box in meters
        # Calculate the image box dimensions and origin from the mesh bounds, adding padding
        image_box = np.array([mesh.bounds[1]-mesh.bounds[0]+image_padding,mesh.bounds[3]-mesh.bounds[2]+image_padding,mesh.bounds[5]-mesh.bounds[4]+image_padding])
        image_origin = np.array([mesh.bounds[0]-image_padding/2,mesh.bounds[2]-image_padding/2,mesh.bounds[4]-image_padding/2])
    
        # Ensure the image shape in each dimension is odd for FFT purposes
        print('\nImage shape (original)  ', np.array(image_box/image_resolution).astype('int'))
        image_shape = make_odd(np.array(image_box/image_resolution).astype('int'))
        print('\nImage shape (odd)       ', image_shape)

    # Create a new image data object and set its dimensions, origin, and spacing
    image = pv.ImageData()
    image.dimensions = image_shape
    image.origin = image_origin
    image.spacing = np.tile(image_resolution,3)

    # Sample the mesh data onto the image grid
    image = image.sample(mesh)

    # Reshape and extract relevant data arrays from the image object for velocity (U) and RST
    velocity_npy = np.reshape(image['U'],(image.dimensions+(image['U'].shape[-1],)), order='F')
    
    # Initialize the maximum velocity variable.
    vmax = np.abs(velocity_npy).max()
    vmean = (velocity_npy).mean()
    print("Mean velocity in field  ", vmean)
    print("Max absolute velocity in field  ", vmax)
    
    magnitude_npy = np.reshape(image['vtkValidPointMask'],(image.dimensions),order='F')

    # Initialize an array to hold synthetic MRI data
    Skv = np.zeros(velocity_npy.shape).astype('complex')
    kv_ = [] # List to store kv values for each direction
    
    # Simulate MRI phase contrast and magnitude images based on velocity and RST data
    for direction,enc_dir in enumerate(enc_dirs):
        for nkv, kv in enumerate(kv_rec[1:,direction]): 
            kv_.append(kv) # Store the current kv value for later reference

            # Calculate the phase shift induced by the fluid velocity (U).
            # This uses the principle of phase contrast MRI, where the phase of the signal
            # encodes the velocity of moving spins (fluid particles in this context).
            # The dot product of velocity vectors with the encoding direction, scaled by kv,
            # gives the phase shift for each voxel. The exponential of 1j times this value
            # converts it into a complex phase factor.  
            phase = np.reshape(np.exp(1j*kv* np.reshape(velocity_npy, (-1,velocity_npy.shape[-1]),
                                                         order='F').dot(enc_dir)), velocity_npy.shape[:3], order='F')
            
            # Skv[...,direction*len(venc)+nkv] = magnitude_npy*signal_loss*phase
            Skv[..., direction*len(venc)+nkv] = magnitude_npy*phase

    ### add complex noise BEFORE cropping
    if add_complex_noise:
        # Introduce noise to the simulated MRI data
        sigma = 0.038/2
        # complex_noise = np.random.normal(0, vmax/vnr, Skv.shape) + \
        #                 np.exp(1j*np.random.normal(0, vmax/vnr, Skv.shape))
        complex_noise = np.random.normal(0, sigma, Skv.shape) + \
                        np.exp(1j*np.random.normal(0, sigma, Skv.shape))
        
        Skv += complex_noise
    
    # Perform downsampling of MR data in fourier domain
    int_complex = []
    for direction in range(Skv.shape[-1]):
        try:
            int_complex = np.concatenate((int_complex,interpolate_3d_grid(Skv[...,direction],
                                                                          reduction_factor, method=method,
                                                                          axes=(0,1,2),kwindow=kwindow,
                                                                          alpha=0.99)[...,np.newaxis]),axis=-1)
        except:
            int_complex = interpolate_3d_grid(Skv[...,direction],reduction_factor, 
                                              method=method,axes=(0,1,2),kwindow=kwindow,
                                              alpha=0.99)[...,np.newaxis]
      
    # HR data 
    hr_dx = np.expand_dims(np.tile(image_resolution,3)*1000,axis=0)
    hr_u = np.expand_dims(velocity_npy[...,0],axis=0)
    hr_v = np.expand_dims(velocity_npy[...,1],axis=0)
    hr_w = np.expand_dims(velocity_npy[...,2],axis=0)
    
    lr_mask = np.expand_dims(np.real(interpolate_3d_grid(magnitude_npy,reduction_factor, method=method,axes=(0,1,2),kwindow=kwindow,alpha=0.99)),axis=0)
    
    mag_u = np.expand_dims(np.abs(int_complex)[...,0],axis=0)
    mag_v = np.expand_dims(np.abs(int_complex)[...,1],axis=0)
    mag_w = np.expand_dims(np.abs(int_complex)[...,2],axis=0)    

    # LR data
    lr_u = np.expand_dims(np.angle(int_complex)[...,0],axis=0)/kv_[0]
    lr_v = np.expand_dims(np.angle(int_complex)[...,1],axis=0)/kv_[1]
    lr_w = np.expand_dims(np.angle(int_complex)[...,2],axis=0)/kv_[2]
    
    if add_complex_noise==False:
        sigma = 0.038/2
        # noise = np.random.normal(0, vmax/vnr, lr_u.shape)
        noise = np.random.normal(0, sigma, lr_u.shape)

        lr_u += noise
        lr_v += noise
        lr_w += noise

    # analyze shift
    
    ########################################################################################################
    # visualize
    
    z_slice =  25 # 28 # 25
    
    plt.contourf(lr_u[0,:,:,z_slice], levels=100, cmap='viridis')
    plt.show()

    ### magniude
    # plt.contourf(mag_u[0,:,:,z_slice], levels=100, cmap='viridis')
    # plt.show()
    
    # plot velocities
    y_slice = 26 # 26
    
    # upsample lr 
    lr_u_int = np.angle(interpolate_3d_grid(int_complex[...,0],mult_factor=2, method='linear',axes=(0,1,2),kwindow=kwindow,alpha=0.99))
    lr_v_int = np.angle(interpolate_3d_grid(int_complex[...,1],mult_factor=2, method='linear',axes=(0,1,2),kwindow=kwindow,alpha=0.99))
    lr_w_int = np.angle(interpolate_3d_grid(int_complex[...,2],mult_factor=2, method='linear',axes=(0,1,2),kwindow=kwindow,alpha=0.99))
    
    # direct comparison
    lr_probe = lr_w[0,:,y_slice,z_slice]
    hr_probe = hr_w[0,:,int(y_slice/reduction_factor),int(z_slice/reduction_factor)]
    lr_int_probe = lr_w_int[:,int(y_slice/reduction_factor),int(z_slice/reduction_factor)]
    
    plt.plot(np.arange(len(lr_probe))*image_resolution/reduction_factor, lr_probe, label='lr_probe')
    # plt.plot(np.arange(len(hr_probe))*image_resolution-2*image_resolution, hr_probe, label='hr_probe')
    plt.plot(np.arange(len(hr_probe))*image_resolution, hr_probe, label='hr_probe')
    plt.plot(np.arange(len(lr_int_probe))*image_resolution, lr_int_probe, label='lr_probe (up-linear)')
    plt.ylabel('velocity [m/s]')
    plt.xlabel('position')
    plt.legend()
    plt.show()
    
    ########################################################################################################
    # Save high-resolution CFD data
    target_file = f'{target_path}data_HR_r{int(image_resolution*10000)}.h5'
    if os.path.exists(target_file):
        os.remove(target_file)
    file = h5py.File(target_file,'a')
    file.create_dataset('dx', data = hr_dx, dtype='half')
    file.create_dataset('u',  data = hr_u, dtype='half')
    file.create_dataset('v',  data = hr_v, dtype='half')
    file.create_dataset('w',  data = hr_w, dtype='half')
    file.create_dataset('u_max', data = np.expand_dims(np.abs(hr_u).max(),axis=0), dtype='half')
    file.create_dataset('v_max', data = np.expand_dims(np.abs(hr_v).max(),axis=0), dtype='half')
    file.create_dataset('w_max', data = np.expand_dims(np.abs(hr_w).max(),axis=0), dtype='half')
    file.create_dataset('mask',  data = np.expand_dims(magnitude_npy,axis=0), dtype='half')
    file.close()
    
    # save vtk file
    fields = ['u', 'v', 'w']
    grid = pv.ImageData()
    grid.dimensions = [x_ for x_ in image.dimensions]
    for i in range(3):
        grid[fields[i]] = np.reshape( velocity_npy[...,i].astype('double'),-1,order='F')
    grid['mask'] = np.reshape( magnitude_npy.astype('double'),-1,order='F')
    grid.save(os.path.join(target_path, f'{target_path}data_HR_r{int(image_resolution*10000)}.vtk'))
    

    ########################################################################################################
    
    target_file = f'{target_path}data_LR_vnr{vnr}_{volume_index}.h5'
    if os.path.exists(target_file):
        os.remove(target_file)
    file = h5py.File(target_file,'a')
    file.create_dataset('dx', data = np.tile(image_resolution,3)*1000/reduction_factor, dtype='half')
    file.create_dataset('venc_u', data = np.expand_dims(np.pi/kv_[0],axis=0), dtype='half')
    file.create_dataset('venc_v', data = np.expand_dims(np.pi/kv_[1],axis=0), dtype='half')
    file.create_dataset('venc_w', data = np.expand_dims(np.pi/kv_[2],axis=0), dtype='half')
    file.create_dataset('mag_u', data = mag_u, dtype='half')
    file.create_dataset('mag_v', data = mag_v, dtype='half')
    file.create_dataset('mag_w', data = mag_w, dtype='half')
    file.create_dataset('mask', data = lr_mask, dtype='half')
    file.create_dataset('u', data = lr_u, dtype='double')
    file.create_dataset('v', data = lr_v, dtype='double')
    file.create_dataset('w', data = lr_w, dtype='double')
    file.close()

    # save vtk file
    fields = ['u', 'v', 'w']
    grid = pv.ImageData()
    grid.dimensions = int_complex[...,0].shape
    grid['u'] = np.reshape(lr_u.astype('double'), -1, order='F')
    grid['v'] = np.reshape(lr_v.astype('double'), -1, order='F')
    grid['w'] = np.reshape(lr_w.astype('double'), -1, order='F')
    grid['mask'] = np.reshape( np.real(interpolate_3d_grid(magnitude_npy,reduction_factor, method=method,axes=(0,1,2),kwindow=kwindow,alpha=0.99)).astype('double'),-1,order='F')
    grid.save(os.path.join(target_path, f'{target_path}data_LR_vnr{vnr}_{volume_index}.vtk'))

print("\nDone")