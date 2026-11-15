'''
TODO : generate pipeline with magnitude signal S_0 computed from velocity field


'''
import h5py
import pyvista as pv
import numpy as np 
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt 
import sys,os,glob,re

from fileOperators import fetchFiles
from gridOperators import make_odd, distance1D, interpolate_3d_grid
from fourierOperators import * 
from functionsRST import functionsRST
 
    
vmax = 0.70  # Initialize the maximum velocity variable.
geom = 'geom3'
# Set the signal-to-noise ratio (SNR) and the reduction factor for downsampling.
snr = 20
# Resolution of the image in meters
image_resolution =  0.1e-3 # 1.25e-3  # default 1.25e-3
# multiplier for saving data
image_res_multiplier = 1e6
# Padding around the image box in meters # 10e-3
image_padding = 4e-3  

# downsampling factor
reduction_factor = 0.5
# plot slice 
y_slice = 20 # 14

# magnitude multiplier
multiplier = 1

kvTRSTkv = functionsRST().kvTRSTkv

# base_path = f'/home/jnestmann/applications/superres/{geom}/org_data/'
# base_path =  f'/Users/jonathan/Documents/applications/superres/{geom}/org_data/'
base_path =  f'/Users/jonathan/Documents/applications/superres/ubend_1000_m1/vtk_eval/'

# target_path = f'/home/jnestmann/applications/superres/{geom}/org_data/'
# target_path =  f'/Users/jonathan/Documents/applications/superres/{geom}/time_data_mod/'
target_path = base_path

#create target path if it doesn't exists
if not os.path.exists(target_path):
    os.makedirs(target_path)

# Loop through files matching the pattern '*inlet_*', which are assumed to be inlet flow files.
for i, inlet_file in enumerate(fetchFiles(base_path + '*', ['inlet_'])):
    inlet_ = pv.read(inlet_file)  # Read the mesh data from the file.
    # Calculate the maximum velocity magnitude in the current inlet file and update vmax if it's higher.
    vmax = max(vmax, np.linalg.norm(inlet_['U'], axis=-1).max())

# Repeat the process for files matching the pattern '*outlet_*', assumed to be outlet flow files.
for i, inlet_file in enumerate(fetchFiles(base_path + '*', ['outlet_'])):
    inlet_ = pv.read(inlet_file)  # Read the mesh data from the file.
    # Again, calculate and potentially update the maximum velocity magnitude.
    vmax = max(vmax, np.linalg.norm(inlet_['U'], axis=-1).max())


extra_velocity_margin = 1.1  # Define an extra margin for the peak velocity to ensure coverage.
# Calculate the encoding velocity (venc) with an extra margin and rounding to the nearest 0.5.
print('extra margin of venc  ', vmax * extra_velocity_margin)
venc = np.array([np.ceil(2 * vmax * extra_velocity_margin) / 2])
print("Rounded venc  ", venc)

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
    print(f"\nVolume index: {volume_index}")
    print("Target  ", target)
    # Read the mesh data from the current target file
    mesh = pv.read(target)

    # Remove all cell data attributes from the mesh except for 'U' (velocity vector) and 'RST' (stress tensor)
    for attribute in mesh.cell_data.keys():
        if attribute == 'U' or attribute == 'RST':
            pass
        else:
            mesh.cell_data.remove(attribute)
                  

    # divergence
    '''
    div_mesh = mesh.compute_derivative(scalars="U", divergence=True)
    divergence = div_mesh.cell_data["divergence"]
    mean_val = np.mean(divergence, dtype=np.float64)
    std_dev = np.std(divergence, dtype=np.float64)
    # laplacian distribution  - MAD : mean absolute deviation
    mad = np.median(np.abs(divergence - np.median(divergence)))
    b = (1 / np.sqrt(2)) * mad
    label = r'($\mu$={:.1f}, $\sigma$={:.1f}, b={:.1f})'.format(mean_val, std_dev, b)

    histogram, bin_edges = np.histogram(divergence, bins='auto') #, density=True)# 'auto')
    plt.plot(bin_edges[0:-1], histogram, label=label)  # Bin edges have one extra element
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  
    plt.title("Divergence computed on CFD mesh")
    plt.ylabel('Divergence [1/s]')
    plt.tight_layout()  
    plt.savefig(target_path + '/' + f'divergence_boxplot_cfd.png', dpi=200)
    plt.show()
    '''
    
    # Initialize image parameters only for the first volume
    if volume_index == 0:
        # Calculate the image box dimensions and origin from the mesh bounds, adding padding
        image_box = np.array([mesh.bounds[1]-mesh.bounds[0]+image_padding,mesh.bounds[3]-mesh.bounds[2]+image_padding,mesh.bounds[5]-mesh.bounds[4]+image_padding])
        image_origin = np.array([mesh.bounds[0]-image_padding/2,mesh.bounds[2]-image_padding/2,mesh.bounds[4]-image_padding/2])
        print("image resolution ", image_resolution)
        # Ensure the image shape in each dimension is odd for FFT purposes
        image_shape = make_odd(np.array(image_box/image_resolution).astype('int'))

    # Create a new image data object and set its dimensions, origin, and spacing
    image = pv.ImageData()
    image.dimensions = image_shape
    image.origin = image_origin
    image.spacing = np.tile(image_resolution,3)

    # Sample the mesh data onto the image grid
    image = image.sample(mesh)
    print("HR image resolution shape    ", image.dimensions)
    
    # Reshape and extract relevant data arrays from the image object for velocity (U) and RST
    velocity_npy = np.reshape(image['U'],(image.dimensions+(image['U'].shape[-1],)), order='F')

    # rst_npy = np.reshape(image['RST'],(image.dimensions+(image['RST'].shape[-1],)),order='F')
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
            phase = np.reshape( np.exp(1j*kv* np.reshape( velocity_npy, (-1,velocity_npy.shape[-1]) ,
                                                         order='F').dot(enc_dir)), 
                               velocity_npy.shape[:3], order='F')

            # Calculate signal modulation due to Reynolds stress tensor (RST).
            # This involves computing kv.T . RST . kv for each voxel, which models the effect
            # of intravoxel velocity dispersion on the MRI signal (a decrease in signal
            # magnitude due to dephasing). The signal loss factor is the exponential of
            # the negative half of this value, representing xsignal attenuation.
            
            # var=np.reshape(kvTRSTkv(np.reshape(rst_npy,(-1,rst_npy.shape[-1]),order='F'),enc_dir*kv),rst_npy.shape[:3],order='F')
            # signal_loss = np.exp(-var/2)

            # Combine the base magnitude (from vtkValidPointMask), signal loss, and phase
            # to simulate the final MRI signal for this kv value and direction.
            # This synthetic signal incorporates effects of both velocity and intravoxel
            # velocity dispersion, capturing key aspects of flow dynamics as would be
            # measured by phase contrast MRI.
            
            # Skv[...,direction*len(venc)+nkv] = magnitude_npy*signal_loss*phase
            Skv[..., direction*len(venc)+nkv] = magnitude_npy*phase

    
    # Perform downsampling of MR data in fourier domain
    int_complex = []
    kwindow = 'gaussian'
    method = 'linear'  # fft, linear
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
    
    downsampled_complex = int_complex.copy()    
    
    # add noise to downsampled image - low resolution
    lr_complex_noise = np.zeros_like(int_complex)
    for i in range(int_complex.shape[-1]):
        lr_complex_noise[...,i] = add_complex_noise_to_complex_image(int_complex[...,i], venc, snr)

    # add noise to high resolution image
    hr_complex_noise = np.zeros_like(Skv)
    for i in range(Skv.shape[-1]):
        hr_complex_noise[...,i] = add_complex_noise_to_complex_image(Skv[...,i], venc, snr)

    ### plot slice
    print("\nDownsampled resolution shape ", lr_complex_noise.shape)
    plt.imshow(np.angle(lr_complex_noise)[...,y_slice,:,2])
    plt.show()
    plt.imshow(np.abs(lr_complex_noise)[...,y_slice,:,2])
    plt.show()

    print("\nHigh resolution shape ", hr_complex_noise.shape)    
    plt.imshow(np.angle(hr_complex_noise)[...,int(y_slice*2),:,2])
    plt.show()
    plt.imshow(np.abs(hr_complex_noise)[...,int(y_slice*2),:,2])
    plt.show()
    
    ########################################################################################################
    # Save high-resolution CFD data
    target_file = f'{target_path}/data_HR_r{int(image_resolution*image_res_multiplier)}.h5'
    if os.path.exists(target_file):
        os.remove(target_file)
    file = h5py.File(target_file,'a')
    file.create_dataset('dx',data = np.expand_dims(np.tile(image_resolution,3)*1000,axis=0), dtype='half')
    file.create_dataset('u',data = np.expand_dims(velocity_npy[...,0],axis=0), dtype='half')
    file.create_dataset('v',data = np.expand_dims(velocity_npy[...,1],axis=0), dtype='half')
    file.create_dataset('w',data = np.expand_dims(velocity_npy[...,2],axis=0), dtype='half')
    file.create_dataset('u_max',data = np.expand_dims(np.abs((velocity_npy[...,0])).max(),axis=0), dtype='half')
    file.create_dataset('v_max',data = np.expand_dims(np.abs((velocity_npy[...,1])).max(),axis=0), dtype='half')
    file.create_dataset('w_max',data = np.expand_dims(np.abs((velocity_npy[...,2])).max(),axis=0), dtype='half')
    file.create_dataset('mask',data = np.expand_dims(magnitude_npy,axis=0), dtype='half')
    file.close()
    
    # save vtk file
    fields = ['u', 'v', 'w']
    grid = pv.ImageData()
    grid.dimensions = [x_ for x_ in image.dimensions]
    for i in range(3):
        grid[fields[i]] = np.reshape( velocity_npy[...,i].astype('double'),-1,order='F')
    grid['mask'] = np.reshape( magnitude_npy.astype('double'),-1,order='F')
    grid.save(os.path.join(target_path, f'{target_path}data_HR_r{int(image_resolution*image_res_multiplier)}.vtk'))

    ########################################################################################################
    # Save high-resolution CFD NOISY data
    target_file = f'{target_path}/data_HR_{snr}_r{int(image_resolution*image_res_multiplier)}.h5'
    if os.path.exists(target_file):
        os.remove(target_file)
    file = h5py.File(target_file,'a')
    file.create_dataset('dx',data = np.tile(image_resolution,3)*1000/reduction_factor, dtype='half')
    file.create_dataset('venc_u',data = np.expand_dims(np.pi/kv_[0],axis=0), dtype='half')
    file.create_dataset('venc_v',data = np.expand_dims(np.pi/kv_[1],axis=0), dtype='half')
    file.create_dataset('venc_w',data = np.expand_dims(np.pi/kv_[2],axis=0), dtype='half')
    file.create_dataset('mag_u',data = np.expand_dims(np.abs(hr_complex_noise)[...,0]*multiplier,axis=0), dtype='half')
    file.create_dataset('mag_v',data = np.expand_dims(np.abs(hr_complex_noise)[...,1]*multiplier,axis=0), dtype='half')
    file.create_dataset('mag_w',data = np.expand_dims(np.abs(hr_complex_noise)[...,2]*multiplier,axis=0), dtype='half')
    file.create_dataset('mask',data = np.expand_dims(magnitude_npy,axis=0), dtype='half')
    file.create_dataset('u',data = np.expand_dims(np.angle(hr_complex_noise)[...,0],axis=0)/kv_[0], dtype='double')
    file.create_dataset('v',data = np.expand_dims(np.angle(hr_complex_noise)[...,1],axis=0)/kv_[1], dtype='double')
    file.create_dataset('w',data = np.expand_dims(np.angle(hr_complex_noise)[...,2],axis=0)/kv_[2], dtype='double')
    file.close()



    ########################################################################################################    
    
    target_file = f'{target_path}/data_LR_{snr}_r{int(image_resolution*image_res_multiplier/reduction_factor)}.h5'
    if os.path.exists(target_file):
        os.remove(target_file)
    file = h5py.File(target_file,'a')
    file.create_dataset('dx',data = np.tile(image_resolution,3)*1000/reduction_factor, dtype='half')
    file.create_dataset('venc_u',data = np.expand_dims(np.pi/kv_[0],axis=0), dtype='half')
    file.create_dataset('venc_v',data = np.expand_dims(np.pi/kv_[1],axis=0), dtype='half')
    file.create_dataset('venc_w',data = np.expand_dims(np.pi/kv_[2],axis=0), dtype='half')
    file.create_dataset('mag_u',data = np.expand_dims(np.abs(lr_complex_noise)[...,0]*multiplier,axis=0), dtype='half')
    file.create_dataset('mag_v',data = np.expand_dims(np.abs(lr_complex_noise)[...,1]*multiplier,axis=0), dtype='half')
    file.create_dataset('mag_w',data = np.expand_dims(np.abs(lr_complex_noise)[...,2]*multiplier,axis=0), dtype='half')
    file.create_dataset('mask',data = np.expand_dims(np.real(interpolate_3d_grid(magnitude_npy*multiplier,reduction_factor, method=method,axes=(0,1,2),kwindow=kwindow,alpha=0.99)),axis=0), dtype='half')
    file.create_dataset('u',data = np.expand_dims(np.angle(lr_complex_noise)[...,0],axis=0)/kv_[0], dtype='double')
    file.create_dataset('v',data = np.expand_dims(np.angle(lr_complex_noise)[...,1],axis=0)/kv_[1], dtype='double')
    file.create_dataset('w',data = np.expand_dims(np.angle(lr_complex_noise)[...,2],axis=0)/kv_[2], dtype='double')
    file.close()
    
    
    
    # save vtk file
    fields = ['u', 'v', 'w']
    grid = pv.ImageData()
    grid.dimensions = lr_complex_noise[...,0].shape
    # grid.dimensions = [int(x_*reduction_factor) for x_ in image.dimensions]
    # grid.spacing = [1/reduction_factor, 1/reduction_factor, 1/reduction_factor]
    for i in range(3):
        grid[fields[i]] = np.reshape( np.angle(lr_complex_noise)[...,i]/kv_[i].astype('double'), -1, order='F')
    grid['mask'] = np.reshape( np.real(interpolate_3d_grid(magnitude_npy,reduction_factor, method=method,axes=(0,1,2),kwindow=kwindow,alpha=0.99)).astype('double'),-1,order='F')
    grid.save(os.path.join(target_path, f'{target_path}data_LR_{snr}_r{int(image_resolution*image_res_multiplier/reduction_factor)}.vtk'))


    # # save vtk file
    # fields = ['u', 'v', 'w']
    # grid = pv.ImageData()
    # grid.dimensions = lr_complex_noise[...,0].shape
    # # grid.dimensions = [int(x_*reduction_factor) for x_ in image.dimensions]
    # # grid.spacing = [1/reduction_factor, 1/reduction_factor, 1/reduction_factor]
    # for i in range(3):
    #     grid[fields[i]] = np.reshape( np.angle(downsampled_complex)[...,i]/kv_[i].astype('double'), -1, order='F')
    # grid.save(os.path.join(target_path, f'{target_path}data_LR_{snr}_downsampled_test.vtk'))
    
    
print("\nDone")


# plt.contourf(data['u'][...][0,0,:,:,18], levels=100, cmap='viridis')