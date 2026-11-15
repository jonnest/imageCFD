import sys
import h5py
import numpy as np 
import matplotlib.pyplot as plt 
import pyvista as pv 


def read_vtk_file(file_path):
    """
    Read a VTK file using pyvista and return a NumPy array of the voxel data.
    """
    # Read the VTK file
    mesh = pv.read(file_path)
    
    # Extract the voxel data as a NumPy array
    np_array = mesh.point_data[mesh.point_data.keys()[0]]  # Assuming scalar field
    np_array = np_array.reshape(mesh.dimensions, order='F')  # Reshape to original dimensions
    
    return np_array

def dice_score(image1, image2):
    """
    Compute the Dice score between two binary NumPy arrays.
    """
    # Flatten the images to 1D arrays for comparison
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()
    
    # Compute the intersection (overlap) between the two images
    intersection = np.sum(image1_flat * image2_flat)
    
    # Compute the Dice score
    dice = (2. * intersection) / (np.sum(image1_flat) + np.sum(image2_flat))
    
    return dice


# Replace 'image1.vtk' and 'image2.vtk' with the paths to your VTK files
image1_path = 'path_to_image1.vtk'
image2_path = 'path_to_image2.vtk'

# Read the VTK files as NumPy arrays
image1 = read_vtk_file(image1_path)
image2 = read_vtk_file(image2_path)

# Compute the Dice score
dice = dice_score(image1, image2)
print(f"Dice score: {dice}")