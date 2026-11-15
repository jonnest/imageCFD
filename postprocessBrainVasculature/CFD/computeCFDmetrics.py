import sys,os,re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from computeMetrics import * 


### data folder
w_dir = '/Users/jonathan/Documents/applications/cfd/eccomas_case/CFD/'
case_id = 'geom1'

target = os.path.join(w_dir,case_id)

spline_dir = '/Users/jonathan/Documents/data/deepFlow/segmentations/polar_tinnitus_eccomas/teccomas_v1'

interpolation_grid_name = 'cfd_interpolation_grid_v1.vtu'

interpolation_grid_file = os.path.join(w_dir, interpolation_grid_name)
interpolation_grid = pv.read(interpolation_grid_file)
############################################################################
### eccomas case
# outflow boundary: left, right branch
# v1 outflow : 1, 2
spline_name = 'spline_2.vtk'

# grid resolution
r_pl = 0.0004
save = False
spline_index = 900

############################################################################
spline_string = spline_name.split('.')[0]
spline_path = os.path.join(spline_dir, spline_name)    
spline = pv.read(spline_path)
spline_pts = spline.points

dir_files = os.listdir(target)

volume_files = []
wall_files = []
for file in dir_files:
    # Check if exclude_string is not in the file name
    if 'vessel_wall' not in file:
        if file.endswith('.vtu'):
            # matching_files.append(os.path.join(target,file))
            volume_files.append(file)
    if 'vessel_wall' in file:
        if file.endswith('.vtu'):
            # matching_files.append(os.path.join(target,file))
            wall_files.append(file)        
    

# sort by chronological order
volume_files.sort()
wall_files.sort()

# TODO: comment out
# volume_files = [volume_files[0]]

print(volume_files)
print(wall_files)

############################################################################
### flow rate 
q = []
for file in volume_files: 
    q.append(extractCFDFlowRate(file, spline_pts, spline_index, 
                                r_pl, target=target, interpolation_grid=interpolation_grid, 
                                save=save))


plt.plot(np.array(q)*1e6, 'o')
plt.ylabel(r'flow rate [ml/s]')
plt.xlabel(r'cardiac cycle')
# plt.title(f'flowrate') 
plt.savefig(w_dir + '/' + f'flow_rate_{case_id}_{spline_string}.png', dpi=200)
plt.show()
############################################################################