import numpy as np
import pyvista as pv
import os
import multiprocessing
import time

# Create a directory for phase-averaged data if it doesn't exist
try:
    os.mkdir('PhaseAveraging')
except:
    pass

#Number of cardiac phases per cardiac cycle
n_phases = 100
for phase in np.arange(n_phases):
    # Get the sorted list of file names in the VTK_merged directory
    frames_0 = np.sort(next(os.walk('VTK_merged/'))[2])
    keywordFilter = ['walls_']
    frames_0 = [sent for sent in frames_0 if any(word in sent for word in keywordFilter)]
    frames_0 = [s.replace('aorta_walls_', '') for s in frames_0]
    frames_0 = [s.replace('.vtu', '') for s in frames_0]
    frames_0 = np.sort(list(map(int, frames_0)))

    # Read the base mesh for the current phase
    base_mesh = pv.read('VTK_merged/aorta_' + str(frames_0[phase]) + '.vtu')
    n_cells = base_mesh.n_cells
    U1 = np.zeros((n_cells, 3))
    RST1 = np.zeros((n_cells, 6))
    P1 = np.zeros((n_cells))

    # Number of cycles
    i = len(frames_0) // n_phases
    print('Generating averaged velocities...')
    for k in range(i - 1):
        print('Cycle ', str(k), ' ...')
        dict_phase = pv.read('VTK_merged/aorta_' + str(frames_0[k * n_phases + n_phases + phase]) + '.vtu')
        U1 += dict_phase.cell_arrays['U'] / (i - 1)
        P1 += dict_phase.cell_arrays['p'] / (i - 1)

    # Function to compute phase-averaged Reynolds stresses
    def RSTcfd2phaseavg(RST, U, U_ds):
        RST_mri = RST.copy()
        RST_mri[:, 0] = RST[:, 0] + (U[:, 0] - U_ds[:, 0]) ** 2
        RST_mri[:, 3] = RST[:, 3] + (U[:, 1] - U_ds[:, 1]) ** 2
        RST_mri[:, 5] = RST[:, 5] + (U[:, 2] - U_ds[:, 2]) ** 2
        RST_mri[:, 1] = RST[:, 1] + (U[:, 0] - U_ds[:, 0]) * (U[:, 1] - U_ds[:, 1])
        RST_mri[:, 2] = RST[:, 2] + (U[:, 0] - U_ds[:, 0]) * (U[:, 2] - U_ds[:, 2])
        RST_mri[:, 4] = RST[:, 4] + (U[:, 1] - U_ds[:, 1]) * (U[:, 2] - U_ds[:, 2])
        return RST_mri

    print('Generating averaged Reynolds stresses...')
    for k in range(i - 1):
        print('Cycle ', str(k), ' ...')
        dict_phase = pv.read('VTK_merged/aorta_' + str(frames_0[k * n_phases + n_phases + phase]) + '.vtu')
        RST1 += RSTcfd2phaseavg(dict_phase.cell_arrays['R'], dict_phase.cell_arrays['U'], U1) / i

    # Create a new mesh for the averaged data
    base_mesh_out = pv.UnstructuredGrid()
    base_mesh_out.copy_structure(base_mesh)
    base_mesh_out['U'] = U1
    base_mesh_out['p'] = P1
    base_mesh_out['RST'] = RST1
    base_mesh_out.save('PhaseAveraging/pv_computed_averaged_' + str(phase) + '.vtu')

    # Process wall shear stress
    base_mesh = pv.read('VTK_merged/aorta_walls_' + str(frames_0[phase]) + '.vtu')
    n_cells = base_mesh.n_cells
    WSS1 = np.zeros((n_cells, 3))
    P1 = np.zeros((n_cells, 1))

    print('Generating averaged wall shear stresses...')
    for k in range(i - 1):
        print('Cycle ', str(k), ' ...')
        dict_phase = pv.read('VTK_merged/aorta_walls_' + str(frames_0[k * n_phases + n_phases + phase]) + '.vtu')
        WSS1 += dict_phase.cell_arrays['wallShearStress'] / (i - 1)
        P1 += dict_phase.cell_arrays['p'] / (i - 1)

    base_mesh_out = pv.UnstructuredGrid()
    base_mesh_out.copy_structure(base_mesh)
    base_mesh_out['WSS'] = WSS1
    base_mesh_out['p'] = P1
    base_mesh_out.save('PhaseAveraging/pv_computed_averaged_walls_' + str(phase) + '.vtu')

    # Process inlet quantities
    base_mesh = pv.read('VTK_merged/aorta_inlet_' + str(frames_0[phase]) + '.vtk')
    n_cells = base_mesh.n_cells
    U1 = np.zeros((n_cells, 3))
    P1 = np.zeros((n_cells, 1))

    print('Generating averaged inlet quantities...')
    for k in range(i - 1):
        print('Cycle ', str(k), ' ...')
        dict_phase = pv.read('VTK_merged/aorta_inlet_' + str(frames_0[k * n_phases + n_phases + phase]) + '.vtk')
        U1 += dict_phase.cell_arrays['U'] / (i - 1)
        P1 += dict_phase.cell_arrays['p'] / (i - 1)

    base_mesh_out = pv.UnstructuredGrid()
    base_mesh_out.copy_structure(base_mesh)
    base_mesh_out['U'] = U1
    base_mesh_out['p'] = P1
    base_mesh_out.save('PhaseAveraging/pv_computed_averaged_inlet_' + str(phase) + '.vtu')

    # Process outlet quantities
    base_mesh = pv.read('VTK_merged/aorta_outlet_' + str(frames_0[phase]) + '.vtk')
    n_cells = base_mesh.n_cells
    U1 = np.zeros((n_cells, 3))
    P1 = np.zeros((n_cells, 1))

    print('Generating averaged outlet quantities...')
    for k in range(i - 1):
        print('Cycle ', str(k), ' ...')
        dict_phase = pv.read('VTK_merged/aorta_outlet_' + str(frames_0[k * n_phases + n_phases + phase]) + '.vtk')
        U1 += dict_phase.cell_arrays['U'] / (i - 1)
        P1 += dict_phase.cell_arrays['p'] / (i - 1)

    base_mesh_out = pv.UnstructuredGrid()
    base_mesh_out.copy_structure(base_mesh)
    base_mesh_out['U'] = U1
    base_mesh_out['p'] = P1
    base_mesh_out.save('PhaseAveraging/pv_computed_averaged_outlet_' + str(phase) + '.vtu')
