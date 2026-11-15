# Import necessary libraries
import vtk  # For handling VTK files
import numpy as np  # For numerical operations
import os, sys  # For system operations (unused here)
# from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk  # For conversions between VTK and NumPy
import matplotlib.pyplot as plt  # For plotting (not used in this script)
from numpy.fft import fftshift, ifftshift, fftn, ifftn, fft2, ifft2  # For FFT operations (not used in this script)
import re  # Regular expressions (not used in this script)
import pyvista as pv  # For working with 3D visualization and VTK files
from scipy.spatial.transform import Rotation as R  # For handling 3D rotations
from scipy.interpolate import UnivariateSpline as US  # For spline interpolation


target_path = '/Users/jonathan/Documents/applications/superres/ubend_1000_v0/vtk_eval/cfd'

# Load the mesh file (surface geometry)
mesh = pv.read(target_path + '/cfd_volume_1.vtk')

# Load the velocity field (e.g., PCMRI data or simulation output)
vel_field = pv.read(target_path + '/cfd_wall_1.vtk')

# Optional: Alternative velocity field input (commented out)
# vel_field = pv.read('OF_smg_5ms_WALE_wf_10.vtk')

# Extract the surface geometry from the mesh
meshs = mesh.extract_surface()

# Compute normals at each point on the surface mesh
meshs.compute_normals(inplace=True)

# Extract normals and surface points from the mesh
normals = meshs.point_data['Normals']
mesh_grid = meshs.points  # Coordinates of surface points

# Define the number of sampling points along each normal
n_points = 3

# Define weights for interpolation, emphasizing certain points
I = np.ones(n_points)
I[0] = 0  # Exclude the first point for interpolation
w = np.ones(n_points)
w[0] = 0.1  # Assign a lower weight to the first point

# Velocity encoding scaling factor (specific to PCMRI velocity scaling)
venc = 2 / np.pi

# Material and geometric parameters
order = 2  # Order of the spline interpolation
l = 1e-3  # Length along the normal direction for probing
mu = 3.5e-3  # Dynamic viscosity of blood (Pa·s)

# Define evenly spaced points along the length of the normal vector
length = np.linspace(0, l, n_points)

# Initialize a container for points to probe
points = np.zeros((mesh_grid.shape[0], mesh_grid.shape[1] * n_points))

# For each sampling point along the normal, calculate the coordinates
for i in range(n_points):
    points[:, 0 + i * 3:3 + i * 3] = np.array(mesh_grid - normals * length[i])

# Probe the velocity field at the calculated points
probed_vel = vel_field.probe(points.reshape(-1, 3))

# Extract and scale the velocity components from the probed field
# probed_vel = np.array((
#     probed_vel.point_data['phase_x'],  # Velocity in x-direction
#     probed_vel.point_data['phase_y'],  # Velocity in y-direction
#     probed_vel.point_data['phase_z']   # Velocity in z-direction
# )).T * venc

probed_vel = np.array((
    probed_vel.point_data['U'][...,0],  # Velocity in x-direction
    probed_vel.point_data['U'][...,0],  # Velocity in y-direction
    probed_vel.point_data['U'][...,0]   # Velocity in z-direction
)).T


# Initialize shear stress container
tau = np.zeros((mesh_grid.shape[0], 3))

# Loop over all points on the surface mesh
for i in range(mesh_grid.shape[0]):
    # Align the normal vector at the current point to the z-axis
    r = R.align_vectors(np.array([[0, 0, 1]]), np.array([normals[i]]))[0].inv()
    
    # Transform the velocity vectors to the rotated coordinate frame
    vel = r.inv().apply(probed_vel.reshape(-1, 3 * n_points)[i].reshape(-1, 3))
    
    # Fit splines to the velocity components along the normal direction
    vxprime = US(length, vel[:, 0] * I, w=w, k=order)  # Spline for x-velocity
    vyprime = US(length, vel[:, 1] * I, w=w, k=order)  # Spline for y-velocity
    
    # Compute the gradient of the velocity components along the normal direction
    vxprimedz = np.gradient(vxprime(np.linspace(0, l, 100)), l / 100)[0]
    vyprimedz = np.gradient(vyprime(np.linspace(0, l, 100)), l / 100)[0]
    
    # Transform the gradients back to the global coordinate frame
    tau[i, :] = r.inv().apply(np.array([vxprimedz, vyprimedz, 0])) * mu
    
    # Print progress every 100 iterations
    if i % 100 == 0:
        print('Iteration ', i)

# Add the computed shear stress to the mesh as a new field
meshs['tau'] = tau / 1060  # Normalize by density of blood (1060 kg/m³)

# Save the updated mesh with shear stress to a VTP file
meshs.save(target_path + '/cfd_wss_1.vtk')
