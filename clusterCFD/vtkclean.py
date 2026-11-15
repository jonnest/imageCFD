import os
import sys
import re
import glob
import shutil
import numpy as np

print('Converting to VTK and removing Foam')

# Get the list of folders inside processor0, excluding 'VTK' and 'constant'
last_folder = next(os.walk('processor0/'))[1]
try:
    last_folder.remove('VTK')
except:
    pass
try:
    last_folder.remove('constant')
except:
    pass

# Sort the folders numerically
sort_arg = np.argsort(list(map(float, last_folder)))
last_folder = [last_folder[i] for i in sort_arg]

# Generate the list of target folders for conversion
c_target = []
for item in last_folder:
    c_target.append(item)
c_target.remove(c_target[-1])  # Remove the latest time folder
c_target = [str(i) for i in c_target]
c_target_str = ','.join(c_target)

# Run the foamToVTK command for different sets of fields
bashCommand = f"mpirun.openmpi -np 96 foamToVTK -fields '(wallShearStress p)' -parallel -noLinks -noInternal -noPointValues -excludePatches '(inlet outlet)' -time '{c_target_str}'"
os.system(bashCommand)

bashCommand = f"mpirun.openmpi -np 96 foamToVTK -fields '(U p)' -parallel -noLinks -noInternal -noPointValues -excludePatches '(walls)' -time '{c_target_str}'"
os.system(bashCommand)

bashCommand = f"mpirun.openmpi -np 96 foamToVTK -fields '(U p R)' -parallel -noLinks -noPointValues -excludePatches '(inlet outlet walls)' -time '{c_target_str}'"
os.system(bashCommand)

# Remove unnecessary directories
dirs = glob.glob('processor*/*[!VTK|constant|0]')
last_dirs = glob.glob('processor*/' + str(last_folder[-1]))
dirs_f = [x for x in dirs if x not in last_dirs]

for i in dirs_f:
    shutil.rmtree(i)

# Remove unnecessary VTK directories
dirs_vtk = glob.glob('processor*/VTK/procB*')
for i in dirs_vtk:
    shutil.rmtree(i)

# Force repositioning of mesh at final iteration
for i in range(96):  # Hardcoded number of processors
    src = f'processor{i}/constant/polyMesh/points'
    dst = f'processor{i}/{str(last_folder[-1])}/polyMesh/points'
    shutil.copyfile(src, dst)

    # Update the location line in the points file
    with open(dst, 'r') as file:
        data = file.readlines()

    data[11] = f'location "{str(last_folder[-1])}/polyMesh"; \n'

    with open(dst, 'w') as file:
        file.writelines(data)
