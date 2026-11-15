# h5 file to vtk
import sys
import h5py
import numpy as np 
import matplotlib.pyplot as plt 
import pyvista as pv 
from fileOperators import *


w_dir = '/Users/jonathan/Documents/applications/superres/geom1/time_data/hr_data'
# w_dir = '/Users/jonathan/Documents/applications/superres/geom3/synthetic/'
# w_dir = '/Users/jonathan/Documents/code/4DFlowNet/data/'
# w_dir = '/Users/jonathan/Documents/applications/superres/test/data/'
# w_dir = '/Users/jonathan/Documents/applications/superres/test/result'
# w_dir = '/Users/jonathan/Documents/applications/superres/geom3/output'
# w_dir = '/Users/jonathan/Documents/applications/superres/oliver_case'
# w_dir = '/Users/jonathan/Documents/applications/superres/oliver_case'
# w_dir = '/Users/jonathan/Documents/applications/cfd/eccomas_case/embedded_50ms'


# w_dir = '/Users/jonathan/Documents/applications/superres/geom1/output/'
# w_dir = '/Users/jonathan/Documents/applications/superres/geom1/data_remote/'

# file_name = 'data_LR_16_res_p12_b20'
# file_name = 'healthy-1mm3_sliced_LR'
# file_name = 'healthy-1mm3_sliced_LR_mod_res_p12_b20'
# file_name = 'healthy-1mm3_sliced_LR_bin_res_p12_b20'

# file_name = '0.4mm_50ms_HR_t0'
file_name = 'data_HR_r4_t0'
# file_name = 'healthy-05mm3_sliced_upsampled_x2'
# file_name = 'healthy-1mm3_sliced_LR_mod'
# file_name = 'healthy-1mm3_sliced_upsampled_mod'

loader = loadData()
data = loader.loadh5file(w_dir+'/'+file_name+'.h5')
keys = h5_tree(data)
print(keys)

image_resolution = 1e-3

# fields = ['mask', 'u', 'v']
fields = ['u', 'v', 'w']
# fields = ['mag_u', 'mag_v', 'mag_w', 'u', 'v', 'w']

image = loader.loadFlowByKeys(w_dir+'/'+file_name+'.h5', fields)
print("\nImage shape   ", image.shape)


t = 0
### save vtk file
grid = pv.ImageData()
try:
    grid.dimensions = [x_ for x_ in image.shape[1:]]
    for i in range(len(fields)):
        grid[fields[i]] = np.reshape(data[fields[i]][...][t].astype('double'),-1,order='F') 
except:
    grid.dimensions = [x_ for x_ in image.shape[2:]] 
    for i in range(len(fields)):
        grid[fields[i]] = np.reshape(data[fields[i]][...][0,t,...].astype('double'),-1,order='F') 

# grid.dimensions = [x_ for x_ in data['u'][...].shape] 
# for i in range(len(fields)):
#     grid[fields[i]] = np.reshape(data[fields[i]][...].astype('double'),-1,order='F') 

grid.save(os.path.join(w_dir, file_name + f'.vtk'))


# ### save vtk file
# grid = pv.ImageData()
# try:
#     grid.dimensions = [x_ for x_ in image.shape[1:]] # TODO : change to 1 for normal workflow
# except:
#     grid.dimensions = [x_ for x_ in image.shape[2:]] # TODO : change to 1 for normal workflow
# for i in range(len(fields)):
#     try:
#         grid[fields[i]] = np.reshape(image[i,...].astype('double'),-1,order='F') # [i,...], [i,0,...]
#     except:
#         grid[fields[i]] = np.reshape(image[i,0,...].astype('double'),-1,order='F') # [i,...], [i,0,...]

# grid.save(os.path.join(w_dir, file_name + f'.vtk'))


1/0
####################################################################
#'''
target_file = w_dir+'/'+file_name+'_mod.h5'
if os.path.exists(target_file):
    os.remove(target_file)
file = h5py.File(target_file,'a')
# file.create_dataset('dx',    data = np.tile(image_resolution,3)*1000, dtype='half')
# file.create_dataset('venc_u',data = [data['venc_u'][...][0][0]], dtype='half')
# file.create_dataset('venc_v',data = [data['venc_v'][...][0][0]], dtype='half')
# file.create_dataset('venc_w',data = [data['venc_w'][...][0][0]], dtype='half')
# file.create_dataset('mag_u', data = data['mag_u'][...][:,0,...] , dtype='half')
# file.create_dataset('mag_v', data = data['mag_v'][...][:,0,...] , dtype='half')
# file.create_dataset('mag_w', data = data['mag_w'][...][:,0,...] , dtype='half')
file.create_dataset('mask', data = data['mask'][...][:,0,...] , dtype='half')
file.create_dataset('u',     data = data['u'][...][:,0,...] , dtype='double')
file.create_dataset('v',     data = data['v'][...][:,0,...] , dtype='double')
file.create_dataset('w',     data = data['w'][...][:,0,...] , dtype='double')
file.close()

#'''
####################################################################
'''
threshold = 95

target_file = w_dir+'/'+file_name+'_bin.h5'
if os.path.exists(target_file):
    os.remove(target_file)
file = h5py.File(target_file,'a')
file.create_dataset('dx',    data = np.tile(image_resolution,3)*1000, dtype='half')
file.create_dataset('venc_u',data = [data['venc_u'][...][0][0]], dtype='half')
file.create_dataset('venc_v',data = [data['venc_v'][...][0][0]], dtype='half')
file.create_dataset('venc_w',data = [data['venc_w'][...][0][0]], dtype='half')
file.create_dataset('mag_u', data = np.where(data['mag_u'][...][:,0,...] >= threshold, 1, 0) , dtype='half')
file.create_dataset('mag_v', data = np.where(data['mag_v'][...][:,0,...] >= threshold, 1, 0) , dtype='half')
file.create_dataset('mag_w', data = np.where(data['mag_w'][...][:,0,...] >= threshold, 1, 0) , dtype='half')
file.create_dataset('u',     data = data['u'][...][:,0,...] , dtype='double')
file.create_dataset('v',     data = data['v'][...][:,0,...] , dtype='double')
file.create_dataset('w',     data = data['w'][...][:,0,...] , dtype='double')
file.close()

'''#