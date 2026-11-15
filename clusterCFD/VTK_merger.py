import numpy as np
import vtk
import pyvista as pv
from scipy import interpolate 
from scipy.io import loadmat
from scipy.special import jv as jv
import os
import shutil
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import multiprocessing
import time

# Set the target path and directory
target_path = os.getcwd()
target_dir = os.path.join(target_path, 'VTK_merged')
n_proc = 48

# Create the target directory if it doesn't exist
try:
    os.mkdir(target_dir)
except:
    pass

# Get the list of processor directories
proc_list = np.sort(next(os.walk(target_path))[1])
keywordFilter = ['processor']
proc_list = [sent for sent in proc_list if any(word in sent for word in keywordFilter)]

# Get the list of time frames
frames_0 = np.sort(next(os.walk(os.path.join(target_path, 'processor0/VTK')))[2])
frames_0 = [sent for sent in frames_0 if any(word in sent for word in keywordFilter)]
frames_0 = [s.replace('processor0_', '') for s in frames_0]
frames_0 = [s.replace('.vtk', '') for s in frames_0]
frames_0 = np.sort(list(map(int, frames_0)))

def multiprocessing_mesh_fusion(frames_0, proc_list=proc_list, target_path=target_path, target_dir=target_dir):
    """
    Function to merge meshes from different processors and save the merged files.
    """
    for t_steps in frames_0:
        # Merge internal mesh data
        print('Generating internal mesh dataset for time ', t_steps)
        mesh = pv.UnstructuredGrid()
        for proc in proc_list:
            mesh = mesh + pv.read(os.path.join(target_path, proc, 'VTK', proc + '_' + str(t_steps) + '.vtk'))
        mesh_c = mesh.cell_data_to_point_data()
        for key in mesh.point_arrays.keys():
            mesh[key] = mesh_c[key]
        mesh.save(os.path.join(target_dir, 'aorta' + '_' + str(t_steps) + '.vtu'))

        # Merge wall mesh data
        print('Generating wall mesh dataset for time ', t_steps)
        mesh = pv.UnstructuredGrid()
        for proc in proc_list:
            mesh = mesh + pv.read(os.path.join(target_path, proc, 'VTK/walls', 'walls_' + str(t_steps) + '.vtk'))
        mesh_c = mesh.cell_data_to_point_data()
        for key in mesh.point_arrays.keys():
            mesh[key] = mesh_c[key]
        mesh.save(os.path.join(target_dir, 'aorta_walls' + '_' + str(t_steps) + '.vtu'))

        # Merge outlet mesh data
        print('Generating outlet mesh dataset for time ', t_steps)
        mesh = pv.UnstructuredGrid()
        for proc in proc_list:
            mesh = mesh + pv.read(os.path.join(target_path, proc, 'VTK/outlet', 'outlet_' + str(t_steps) + '.vtk'))
        mesh_c = mesh.cell_data_to_point_data()
        for key in mesh.point_arrays.keys():
            mesh[key] = mesh_c[key]
        mesh.save(os.path.join(target_dir, 'aorta_outlet' + '_' + str(t_steps) + '.vtk'))

        # Merge inlet mesh data
        print('Generating inlet mesh dataset for time ', t_steps)
        mesh = pv.UnstructuredGrid()
        for proc in proc_list:
            mesh = mesh + pv.read(os.path.join(target_path, proc, 'VTK/inlet', 'inlet_' + str(t_steps) + '.vtk'))
        mesh_c = mesh.cell_data_to_point_data()
        for key in mesh.point_arrays.keys():
            mesh[key] = mesh_c[key]
        mesh.save(os.path.join(target_dir, 'aorta_inlet' + '_' + str(t_steps) + '.vtk'))

def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Split the frames into chunks for parallel processing
chunked_frame_0 = list(chunks(frames_0, int(len(frames_0) / n_proc) + 1))
print(chunked_frame_0)

if __name__ == '__main__':
    starttime = time.time()
    processes = []

    # Start parallel processes
    for i in np.arange(0, n_proc):
        p = multiprocessing.Process(target=multiprocessing_mesh_fusion, args=(chunked_frame_0[i],))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    print('That took {} seconds'.format(time.time() - starttime))
