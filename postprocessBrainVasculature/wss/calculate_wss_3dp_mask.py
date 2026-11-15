''' WSS calculator 
Oyre 1998 - paraboloid sector
WSS calculator from Feridan 2020 https://github.com/EdwardFerdian/wss_mri_calculator/tree/master
Base implementation based on Petersson 2012 
"Assessment of the accuracy of MRI wall shear stress estimation using numerical simulations"
https://doi.org/10.1002/jmri.23610
'''

import pyvista as pv
import numpy as np
import logging
import matplotlib.pyplot as plt 
import scipy as scp
import time
import pymeshfix
from mpl_toolkits.mplot3d import Axes3D

from wss_utils import *
from surfaceMesh import *
from gridOperators import *

from scipy.ndimage import binary_erosion
from scipy.optimize import least_squares
from scipy.interpolate import griddata, UnivariateSpline
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from skimage.restoration import denoise_bilateral
from scipy.optimize import curve_fit

# TODO 
# - bifurcations : splines need to be connected or looped over branches

# ------------------------------------------------------------------
# paths and files
# ------------------------------------------------------------------

target_path = '/Users/jonathan/Documents/applications/superres/ubend_1000_m1/vtk_eval'
# target_path = '/Users/jonathan/Documents/applications/superres/geom3/time_data/vtk_eval'

# input_filepath = target_path + '/data_HR_20_r100.h5'

input_filepath = target_path + '/data_HR_r100.h5'
# input_filepath = target_path + '/data_upsampled_20_r200.h5'
# input_filepath = target_path + '/data_LR_20_r100.h5'

# mask file - different for super-resolution upsampled file
# mask_filepath = input_filepath
mask_filepath = target_path + '/data_HR_r100.h5'

voxel_size = 0.1  # in mm

# name with path of cfd folder
cfd_filename = '/cfd/cfd_wss_1.vtk'
# cfd_filename = '/cfd/vessel_wall_49800.vtu'

# save filenames for extracted wall points
wall_vtk_filename = 'wall_point_HR_r100_ang18'
surface_vtk_filename = 'surface_points_ang18'

# ------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------
# compute WSS 
parabolic_sector_fitting = True
# find vessel wall 1d or 3d
find_vessel_wall_3d = True

# maximum diameter to check gradients - in mm
r_sample_max = 3 
# for 3d sector parabolid
r_offset = 1*voxel_size  # 1*voxel_size
r_layer = 3*voxel_size   # 2*voxel_size
# find radius using 3d parabolid
r_layer_find = 3*voxel_size
# negative offset of initial r_min to search for boundary points via gradients
r_min_initial_offset = 4*voxel_size


# find radius - with or without no-slip condition
find_radius_noslip = False
# sector definition : delta phi of sector - # default 20
phi_sector = 18 # 10, 90
delta_phi = phi_sector/180*np.pi
z_tol = voxel_size/2

# all as vector multiplication # 30*voxel_size = max sample radius
nr_radial = int(r_sample_max/voxel_size)
# discretized ngles to analyze - multiple of 360 !
angle_step = 18 # default 18

plot_path = target_path + '/plots'

# input parameters
# Inward normal distance to sample points from wall (in mm)
inward_distance = voxel_size # in mm
# dynamic viscosity
viscosity = 3.18e-3
density = 1060

# show plot
show_plot = False
# Show WSS contours for visualization
show_wss_contours = True
# Save volume mesh and surface WSS in vtk file
save_to_vtk = True
# Prefix for the output file when save_to_vtk is True
vtk_filename = 'result'
# smooth surface
smoothing = 500
# Logging level
loglevel = logging.INFO

# centerline extraction by voxelization
n_pts = 100 # 500, 1000
grid_voxel_size = 0.2 # 0.5

# make dir it doesn't exist
os.makedirs(plot_path, exist_ok=True)
# ------------------------------------------------------------------
### define target point for analysis
# stenosis wall point
cfd_point_stenosis = np.array([-0.000951654, 0, 0.00313039])*1000
# inflow wall point
# cfd_point_stenosis = np.array([-0.00199243, 0, 0.0158897])*1000

target_point_folder = 'stenosis_point'
# make directory
target_point_path = plot_path + '/' + target_point_folder
os.makedirs(target_point_path, exist_ok=True)


### read cfd
cfd_file = target_path + '/' + cfd_filename

# read CFD
cfd_mesh = pv.read(cfd_file)

# translate both meshes to same origin and scaling
cfd_mesh = cfd_mesh.scale(1000)
new_origin = (0,0,0)
translation_cfd = tuple(new_origin[i] - cfd_mesh.center[i] for i in range(3))
# translate mesh and point
cfd_mesh.translate(translation_cfd, inplace=True)
pv.PolyData(cfd_point_stenosis).translate(translation_cfd, inplace=True)

print("Original cfd point   ", cfd_point_stenosis)
print("Translated cfd point ", cfd_point_stenosis)

cfd_target_pts, cfd_target_idx = find_closest_point(cfd_point_stenosis, cfd_mesh.points)

cfd_wss = np.linalg.norm(cfd_mesh.point_data['wallShearStress'],axis=1)*density
# cfd_wss = np.linalg.norm(cfd_mesh.cell_data['wallShearStress'],axis=1)*density

cfd_wss_max = cfd_wss.max()
cfd_wss_mean = cfd_wss.mean()

print("\nCFD WSS max [Pa]    ", cfd_wss_max)
print("CFD WSS mean [Pa]   ", cfd_wss_mean)

# ------------------------------------------------------------------
# surface extraction
# ------------------------------------------------------------------1
# If the dimension is not isotropic, you might as well edit it here
spacing = (voxel_size,voxel_size,voxel_size) # in mm
inward_distance = inward_distance
smoothing_iteration = smoothing

# 1. Load velocity vectors and segmentation/mask
# Note: Please replace the loader depending on your data format, this is just an example
u,v,w = load_my_vector_fields(input_filepath, ['u', 'v', 'w'], 0)
# Convert velocities in mm/s
u,v,w = u*1000, v*1000, w*1000

print("shape u   ", u.shape)
print("shape v   ", v.shape)
print("shape w   ", w.shape)
u,v,w = u.astype(np.float64), v.astype(np.float64), w.astype(np.float64)

mask = load_my_segmentation(mask_filepath, 'mask', 0)
mask = mask.astype(np.float64)

    
# 2. Construct uniform grid for vector values and mask
velocity = create_uniform_vector(u,v,w, spacing)
mesh = create_uniform_grid(mask, spacing)

# 3. Get the volume within the mask (for visualization purpose)
volume = mesh.sample(velocity)
volume.set_active_scalars("Velocity")

# 4. Get the surface of the mesh
# Note: do not use extract_geometry, the PyVista manual on the website is misleading
# surf = mesh.threshold_percent(40).extract_surface()
# Extract the surface of the masked grid
masked_mesh = mesh.threshold(scalars="mask", value=0.5)  # Adjust threshold as needed
surf = masked_mesh.extract_surface()
# surf = mesh.extract_surface()
surf = surf.smooth(n_iter=smoothing_iteration)

# p = pv.Plotter(notebook=0)
# p.add_mesh(surf, show_edges=True, cmap='jet')
# p.show()

# get cross section points
surface = surf.triangulate()
surface = pymeshfix.MeshFix(surface)
surface.repair(joincomp=True, remove_smallest_components=True) 
surface = surface.mesh        
print(f"The surface still has {surface.n_open_edges} open edges.")


pyskeleton, origin, spacing_skeleton, _ = pyskeletonize(surface,grid_voxel_size=grid_voxel_size,
                                                        voxelization_size=grid_voxel_size,
                                                        estimate_distance=True,coarse_voxel_size=grid_voxel_size*3,
                                                        distance_tolerance=1e-3, target=target_path)

tf_branches, tf_gradients = extractTerminalBranches(pyskeleton, origin=origin, spacing=spacing_skeleton,  
                                                    n_p_interp = n_pts, n_o_interp = 3,
                                                    cut_off=1e-2, min_points=0.1, target=target_path)

# resample mesh
surface = uniformlyResampleMesh(surface,n_points=20000,n_subdivisions=2)

# window length for spline smoothing
window_length = int(to_closest_odd(0.2*n_pts))
smoothed_branch = smooth_3d_curve(tf_branches['0'], window_length, polyorder=2)

spline = pv.Spline(smoothed_branch, 1000)
spline.points = spline.points
spline.save(target_path + f'/smooth_spline.vtk')

### get radii with raytracing 
# recompute gradients with smoothed branch
smoothed_gradient = np.zeros(smoothed_branch.shape)
for i in range(len(smoothed_branch)):
    if i==0:
        normal_tmp = smoothed_branch[i] - smoothed_branch[i+1]
        plane_normal = normal_tmp/np.linalg.norm(normal_tmp)
        smoothed_gradient[i,:] = plane_normal
    else:
        normal_tmp = smoothed_branch[i] - smoothed_branch[i-1]
        plane_normal = normal_tmp/np.linalg.norm(normal_tmp)
        smoothed_gradient[i,:] = plane_normal

# compute radii from centerline to surface
# spline_radii = getRadiiOfSurface(surface, smoothed_branch, smoothed_gradient, n_pts)

# ------------------------------------------------------------------
# initialize
# ------------------------------------------------------------------
# set angles to compute wss on
angles = range(0, 360, angle_step)
         
# initialize wall points with wss
wall_points = np.empty((0,3))
wss = np.array([])
wss_planes = np.empty((n_pts,len(angles)))

# surface points
surface_points = np.zeros((n_pts, len(angles), 3))

# ------------------------------------------------------------------
# loop over all center points
for plane_idx in range(n_pts):
# for plane_idx in range(10):
    t1 = time.time()
    # inflow 10/500
    # stenosis 350/500
    # stenosis 74/100 for branc 0-0 and 1-1
    # plane_idx = 90 # 70 # 30, 65, 70 
    
    ### analysis in rotated reference coordinate system and polar coordinates
    # get normal vector of current plane with point origin 
    # smoothed_gradient or tf_branches['0'], tf_gradients['0'],
    # plane_normal = smoothed_gradient[plane_idx]/np.linalg.norm(smoothed_gradient[plane_idx])
    if plane_idx==0:
        normal_tmp = smoothed_branch[plane_idx] - smoothed_branch[plane_idx+1]
        plane_normal = normal_tmp/np.linalg.norm(normal_tmp)
    else:
        normal_tmp = smoothed_branch[plane_idx] - smoothed_branch[plane_idx-1]
        plane_normal = normal_tmp/np.linalg.norm(normal_tmp)
        
    plane_origin = smoothed_branch[plane_idx]
    
    # rotation all points for x,y plane and normal is z-axis
    euler_rotations = align_vectors(plane_normal, [0,0,1])
    volume_rot = volume.copy()
    volume_rot = volume_rot.rotate_x(euler_rotations[0],point=plane_origin, transform_all_input_vectors=True)
    volume_rot = volume_rot.rotate_y(euler_rotations[1],point=plane_origin, transform_all_input_vectors=True)
    volume_rot = volume_rot.rotate_z(euler_rotations[2],point=plane_origin, transform_all_input_vectors=True)
    
    # loop over phi and get velocity curve in each radial direction
    inplane_vectors = generate_vectors_in_plane(plane_normal, angle_step=angle_step)

    ### raytracing to find radii   
    initial_radii = np.zeros(len(inplane_vectors))
    for i in range(len(inplane_vectors)):
        # ray trace for each inplane vector
        bnd_tmp = surface.multi_ray_trace(np.expand_dims(plane_origin, 0), 
                                          np.expand_dims(inplane_vectors[i], 0), 
                                          first_point=True, retry=True)[0]

        initial_radii[i] = np.linalg.norm(plane_origin - bnd_tmp)
        # store surface points of raytracing
        surface_points[plane_idx, i, :] = bnd_tmp
    
    
    # number of interpolation points
    dr = voxel_size * np.arange(0,nr_radial)
    
    # initialize maximum radius array 
    radii = np.empty(len(inplane_vectors))
    
    ### 2 parameter optimization problem
    ### find largest radius with highest gradient
    
    ### are NOT in rotated coordinate system
    # create array of all considered plane points
    # inplane_pts = (angles, radials, 3)
    inplane_pts = np.empty((len(angles),nr_radial,3))
    for j, step in enumerate(dr):
        inplane_pts[:,j,:] = plane_origin + step*inplane_vectors

    # rotate inplane points
    inplane_pts_array = pv.PolyData(np.reshape(inplane_pts, (-1,3)))
    inplane_pts_array = inplane_pts_array.rotate_x(euler_rotations[0],point=plane_origin, transform_all_input_vectors=True)
    inplane_pts_array = inplane_pts_array.rotate_y(euler_rotations[1],point=plane_origin, transform_all_input_vectors=True)
    inplane_pts_array = inplane_pts_array.rotate_z(euler_rotations[2],point=plane_origin, transform_all_input_vectors=True)
    
    inplane_pts_rot = np.reshape(inplane_pts_array.points, inplane_pts.shape)
    # plt.plot(inplane_pts[:,:,2].ravel())    
    # plt.show()
    
    inplane_pts_pv = pv.PolyData(inplane_pts_array).sample(volume_rot)
    
    # NOTE : when computing the magnitude, the sign is lost
    inplane_vel_array = np.vstack((inplane_pts_pv['u'], inplane_pts_pv['v'], inplane_pts_pv['w'])).T
    inplane_mag = np.linalg.norm(inplane_vel_array, axis=1)      
    inplane_mag = np.reshape(inplane_mag, inplane_pts.shape[:-1])
    
    # get radial distances and angles of points
    translated_points = volume_rot.points - plane_origin
    r = np.linalg.norm(translated_points[:, :2], axis=1)  # Radial distances (x, y components only)
    phi = np.arctan2(translated_points[:, 1], translated_points[:, 0])  # Angles in radians
    phi = np.mod(phi, 2 * np.pi)

    print("phi shape ", phi.shape)
    
    translated_inplane_pts = inplane_pts_pv.points - plane_origin
    z_plane = np.mean(translated_inplane_pts[...,2])

    # init 
    wss_plane = []
    if find_vessel_wall_3d:
        # loop over all phi and radial direction vectors
        for idx, angle in enumerate(angles):
            if len(inplane_vectors) != len(angles):
                raise ValueError('Number of angles and vectors does not match')
            
            gradients = []
            # loop over all possible r_max, first one should be able to have well-defined 3dParabolid
            # TODO : initial radius
            # r_min = r_offset + r_layer_find
            r_min = initial_radii[idx] - r_min_initial_offset
            
            for r_max in np.arange(r_min, r_sample_max, voxel_size):
                # get sector points
                phi_center = angle/180*np.pi
                # Define the sector bounds based on phi_center and delta_phi
                phi_min = phi_center - delta_phi / 2
                phi_max = phi_center + delta_phi / 2
                
                # Handle angle wrapping around 0 and 2*pi
                if phi_min < 0:
                    phi_mask = (phi >= phi_min + 2 * np.pi) | (phi <= phi_max)
                elif phi_max > 2 * np.pi:
                    phi_mask = (phi >= phi_min) | (phi <= phi_max - 2 * np.pi)
                else:
                    phi_mask = (phi >= phi_min) & (phi <= phi_max)

                # Step 6: Create a mask for radial distance
                r_mask = (r >= r_max-(r_layer_find+r_offset)) & (r <= r_max-r_offset)

                z_mask = (translated_points[:, 2] >= z_plane - z_tol) & (translated_points[:, 2] <= z_plane + z_tol)
                # print("number z mask selected ", np.count_nonzero(z_mask))
                
                # Step 7: Combine masks to find points inside the sector
                sector_mask = phi_mask & r_mask & z_mask
                # print("number sector points ", np.count_nonzero(sector_mask))
                
                # get points in rotated volume
                points_sector = volume_rot.points[sector_mask]
                # get velocities
                u_sector = volume_rot['u'][sector_mask]
                v_sector = volume_rot['v'][sector_mask]
                w_sector = volume_rot['w'][sector_mask]
                vel_sector = np.vstack((u_sector, v_sector, w_sector)).T
                mag_sector = np.linalg.norm(vel_sector, axis=1)
                
                # get single wall point of sector #r_max*inplane_vectors[idx] 
                # wall point in rotated system
                wall_point_inplane = plane_origin + [r_max*np.cos(phi_center), r_max*np.sin(phi_center), 0]
                        
                # add wall point
                points_sector = np.concatenate(([wall_point_inplane,], points_sector))

                wall_point_inplane_pv = pv.PolyData(wall_point_inplane).sample(volume_rot)
                vel_wall_point_inplane = np.vstack((wall_point_inplane_pv['u'], wall_point_inplane_pv['v'], wall_point_inplane_pv['w'])).T
                mag_wall_point_inplane = np.linalg.norm(vel_wall_point_inplane, axis=1)
                
                ### magnitude - wall point - no slip or velocity
                if find_radius_noslip == True:
                    mag_sector = np.concatenate(([0,], mag_sector))
                else:
                    mag_sector = np.concatenate(([mag_wall_point_inplane[0],], mag_sector))
            
                # Initial guesses for a, b, c, d
                initial_guess = [1, 1, 1, 1]
                result = least_squares(residuals_paraboloid, initial_guess, 
                                    args=(points_sector[:,0], points_sector[:,1], mag_sector))
                # Extract the optimized parameters
                params_parabolid = result.x
                a_opt, b_opt, c_opt, d_opt = params_parabolid
                # print(f"Optimized parameters: a={a_opt}, b={b_opt}, c={c_opt}, d={d_opt}")

                # get gradient on wall and get WSS
                du_mag = derivate_magnitude(wall_point_inplane, params_parabolid)
                # save gradients
                gradients.append(du_mag)
                
            ### plot gradient over radius
            # plt.plot(np.arange(r_min, r_sample_max, voxel_size), gradients)
            # plt.xlabel('radial distance [mm]')
            # plt.ylabel('magnitude gradient')
            # plt.show()
            
            # get largest gradient and its location
            radii[idx] = np.arange(r_min, 3, voxel_size)[np.argmax(gradients)]
    
    def parabolic_func(x, a, b, c):
        return a * x**2 + b * x + c
    
    def compute_rmse(y_true, y_fit):
        return np.sqrt(np.mean((y_true - y_fit)**2))

    # find vessel wall with 1d interpolation of velocity magnitude
    if find_vessel_wall_3d == False:
        r_cs_idx = []
        r_cs = []
        r_cs_ = []
        
        # # loop over all phi and radial direction vectors
        # for idx, angle in enumerate(angles):
        #     if len(inplane_vectors) != len(angles):
        #         raise ValueError('Number of angles and vectors does not match')
            
        #     # Apply Gaussian filter for noise reduction
        #     sigma_mag = 2  # Smoothing parameter for Gaussian filter
        #     # filtered_mag = gaussian_filter1d(inplane_mag[idx], sigma_mag)
        
        #     # Optional: Use bilateral filtering for edge-preserving smoothing
        #     # (Requires scaling data between 0 and 1 if using skimage)
        #     # filtered_mag = denoise_bilateral(inplane_mag[idx], sigma_color=0.1, sigma_spatial=0.1)
        #     filtered_mag = inplane_mag[idx, ...]
            
        #     ### find largest gradient 
        #     ### TODO : alternative - iterate from center out to find highest grad
        #     # fit univariate spline
        #     spline = UnivariateSpline(dr, filtered_mag)  # s=10
        #     dx_spline = spline.derivative()
            
        #     # high res evaluation of spline
        #     dr_ = voxel_size * np.arange(0, nr_radial, 0.01)
        #     spline_grads = dx_spline(dr_)
            
        #     # find max gradient idx
        #     max_grad_idx_ = np.argmax(np.abs(spline_grads))
        #     max_grad_dr_ = dr_[max_grad_idx_]
        #     max_grad_ = spline_grads[max_grad_idx_]
            
        #     radii[idx] = max_grad_dr_
            
        #     ### plot spline
        #     plt.plot(dr, inplane_mag[idx,:], 'b')
        #     plt.plot(dr_, spline(dr_), 'r')
        #     plt.plot(max_grad_dr_, 0, 'o')
        #     plt.show()
            
        #     1/0
        #     # get max_grad index of coarse dr
        #     r = np.argmin(np.abs(dr - max_grad_dr_))
        #     r_cs_idx.append(r)
        #     r_cs_.append(max_grad_dr_)
        #     r_cs.append(dr[r])
        #     radii[idx] = max_grad_dr_
        
        # loop over all phi and radial direction vectors
        for idx, angle in enumerate(angles):
            if len(inplane_vectors) != len(angles):
                raise ValueError('Number of angles and vectors does not match')
            
            # Apply Gaussian filter for noise reduction
            sigma_mag = 2  # Smoothing parameter for Gaussian filter
            # filtered_mag = gaussian_filter1d(inplane_mag[idx], sigma_mag)
        
            # Optional: Use bilateral filtering for edge-preserving smoothing
            # (Requires scaling data between 0 and 1 if using skimage)
            # filtered_mag = denoise_bilateral(inplane_mag[idx], sigma_color=0.1, sigma_spatial=0.1)
            filtered_mag = inplane_mag[idx, ...]
            
            # Modified Code
            errors = []  # To track RMSE error
            best_fit_points = []  # Track best configuration

            # Initialize variables
            num_points = len(dr)
            regression_points = 3
            current_points = []  # Store iteratively added points
            current_errors = []

            # Start iterative process
            for i in range(3, num_points):  # Start with 3 points, add one at a time
            # for i in range(len(dr) - regression_points + 1):
                # Add points starting from the center (beginning of dr) and extending outward
                
                points_idx = np.arange(0, i)  # Take the first i points
                # points_idx = np.arange(i, i + regression_points)  # Window of 3 points
                
                current_points = dr[points_idx]
                current_mag = filtered_mag[points_idx]

                # Perform parabolic regression
                params, _, curve_dict,_,_ = curve_fit(parabolic_func, current_points, current_mag, full_output=True)
                fit_values = parabolic_func(dr, *params)

                # Compute error (RMSE)
                # rmse = compute_rmse(filtered_mag, fit_values)
                rmse = np.linalg.norm(curve_dict['fvec'])
                current_errors.append(rmse)

                # Store configuration with the minimum error
                if len(errors) == 0 or rmse < min(errors):
                    best_fit_points = current_points
                    best_params = params

                # Track errors for plotting
                errors.append(rmse)

                # Plot current fit
                plt.figure(figsize=(8, 4))
                plt.plot(dr, filtered_mag, 'b', label='Original Data')
                plt.plot(dr, fit_values, 'r--', label=f'Current Fit ({i} points)')
                plt.scatter(current_points, current_mag, color='g', label='Selected Points')
                plt.title(f'Current Fit with {i} Points')
                plt.legend()
                plt.show()
    
            # Plot RMSE over added points
            plt.figure(figsize=(8, 4))
            # plt.plot(range(3, max_points + 1), errors, '-o')
            plt.plot(range(len(errors)), errors, '-o')
            plt.xlabel('Number of Points Used')
            plt.ylabel('RMSE')
            plt.title('Fitting Error over Number of Points')
            plt.grid(True)
            plt.show()

            # Final Best Fit Plot
            if best_params is not None:
                final_fit = parabolic_func(dr, *best_params)
                plt.figure(figsize=(8, 4))
                plt.plot(dr, filtered_mag, 'b', label='Original Data')
                plt.plot(dr, final_fit, 'r--', label='Best Parabolic Fit')
                plt.scatter(best_fit_points, parabolic_func(best_fit_points, *best_params), color='g', label='Best Fit Points')
                plt.title('Final Best Parabolic Fit')
                plt.legend()
                plt.show()

            1/0
            ### find largest gradient 
            ### TODO : alternative - iterate from center out to find highest grad
            # fit univariate spline
            spline = UnivariateSpline(dr, filtered_mag)  # s=10
            dx_spline = spline.derivative()
            
            # high res evaluation of spline
            dr_ = voxel_size * np.arange(0, nr_radial, 0.01)
            spline_grads = dx_spline(dr_)
            
            # find max gradient idx
            max_grad_idx_ = np.argmax(np.abs(spline_grads))
            max_grad_dr_ = dr_[max_grad_idx_]
            max_grad_ = spline_grads[max_grad_idx_]
            
            radii[idx] = max_grad_dr_
            
            ### plot spline
            plt.plot(dr, inplane_mag[idx,:], 'b')
            plt.plot(dr_, spline(dr_), 'r')
            plt.plot(max_grad_dr_, 0, 'o')
            plt.show()
            
            1/0
            # get max_grad index of coarse dr
            r = np.argmin(np.abs(dr - max_grad_dr_))
            r_cs_idx.append(r)
            r_cs_.append(max_grad_dr_)
            r_cs.append(dr[r])
            radii[idx] = max_grad_dr_
        
    # smooth 
    sigma = np.sqrt(np.var(radii))*4
    # extend array and make periodic
    padding = int(len(radii)/4)
    extended_radii = np.concatenate([radii[-padding:], radii, radii[:padding]])
    # extended_smoothed_radii = gaussian_filter1d(extended_radii, sigma=sigma)  
    # extended_smoothed_radii = denoise_bilateral(extended_radii, sigma_color=0.1, sigma_spatial=4)  
    extended_smoothed_radii = savgol_filter(extended_radii, window_length=7, polyorder=5)
    radii_smoothed = extended_smoothed_radii[padding:-padding]
    
    ### fit ellipse to data
    # Initial guess for a and b (semi-major and semi-minor axes)
    initial_guess = [max(radii), min(radii)]
    initial_guess = [1 if x == 0 else x for x in initial_guess]
    
    # Perform the least squares fitting
    result = least_squares(residuals_ellipse, initial_guess, args=(angles, radii))
    # Get the fitted parameters (a and b)
    a_fit, b_fit = result.x
    # Recompute the radii using the fitted ellipse parameters
    radii_ellipse = ellipse_radii([a_fit, b_fit], angles)

    # spline fit 
    # dr_spline = UnivariateSpline(angles, radii)  # s=10
    # extended_angles = range(-angle_step*padding, 360+angle_step*padding, angle_step)
    # dr_spline_ex = UnivariateSpline(extended_angles, extended_radii)  # s=10
    dr_spline = UnivariateSpline(angles, radii)  # s=10
    
    # coordinates
    x_original = radii * np.cos(angles)
    y_original = radii * np.sin(angles)
    
    x_ellipse = radii_ellipse * np.cos(angles)
    y_ellipse = radii_ellipse * np.sin(angles)
    
    x_spline = dr_spline(angles) * np.cos(angles)
    y_spline = dr_spline(angles) * np.sin(angles)
    
    x_smoothed = radii_smoothed * np.cos(angles)
    y_smoothed = radii_smoothed * np.sin(angles)
    
    # plt.scatter(x_original, y_original, label='org')
    # plt.scatter(x_ellipse, y_ellipse, label='ellipse')
    # plt.scatter(x_smoothed, y_smoothed, label='smoothed')
    # plt.scatter(x_spline, y_spline, label='spline')
    # plt.legend()
    # plt.show()
    
    # plt.plot(angles, radii, label='org')
    # plt.plot(angles, radii_ellipse, label='ellipse')
    # plt.plot(angles, radii_smoothed, label='smoothed')
    # plt.plot(angles, dr_spline(angles), label='spline')
    # plt.xlabel('angle')
    # plt.ylabel('maximum radius [mm]')
    # plt.legend()
    # plt.show()
    
    
    ### TODO : choose smooth radii array
    # radii_smoothed = fitted_radii
    # radii_smoothed = radii
    radii_smoothed = dr_spline(angles)
    
    if plane_idx % 10 == 0:
        plt.plot(radii)
        plt.plot(radii_smoothed)
        plt.xlabel('radius id')
        plt.ylabel('maximum radius [mm]')
        # plt.show()
        plt.savefig(plot_path + '/' + f'radii_plane{plane_idx}.png', dpi=200)

    # Compute WSS using 3dP fit
    if parabolic_sector_fitting:
        for idx, angle in enumerate(angles):     
            # get sector points
            phi_center = angle/180*np.pi
            # Define the sector bounds based on phi_center and delta_phi
            phi_min = phi_center - delta_phi / 2
            phi_max = phi_center + delta_phi / 2
        
            # maximum radius for sector of smoothed radius curve
            r_max = radii_smoothed[idx]   
            
            # Handle angle wrapping around 0 and 2*pi
            if phi_min < 0:
                phi_mask = (phi >= phi_min + 2 * np.pi) | (phi <= phi_max)
            elif phi_max > 2 * np.pi:
                phi_mask = (phi >= phi_min) | (phi <= phi_max - 2 * np.pi)
            else:
                phi_mask = (phi >= phi_min) & (phi <= phi_max)

            # Step 6: Create a mask for radial distance
            r_mask = (r >= r_max-(r_layer+r_offset)) & (r <= r_max-r_offset)

            z_mask = (translated_points[:, 2] >= z_plane - z_tol) & (translated_points[:, 2] <= z_plane + z_tol)
            
            # Step 7: Combine masks to find points inside the sector
            sector_mask = phi_mask & r_mask & z_mask
            
            # get points in rotated volume
            points_sector = volume_rot.points[sector_mask]
            # get velocities
            u_sector = volume_rot['u'][sector_mask]
            v_sector = volume_rot['v'][sector_mask]
            w_sector = volume_rot['w'][sector_mask]
            vel_sector = np.vstack((u_sector, v_sector, w_sector)).T
            mag_sector = np.linalg.norm(vel_sector, axis=1)
            
            # get single wall point inplane of sector #r_max*inplane_vectors[idx] 
            wall_point_inplane = plane_origin + [r_max*np.cos(phi_center), r_max*np.sin(phi_center), 0]
            wall_point = plane_origin + r_max * inplane_vectors[idx] 

            # plt.scatter(points_sector[:,0], points_sector[:,1], alpha=0.1)
            # plt.plot(plane_origin[0], plane_origin[1], 'o', color='blue')
            # plt.plot(wall_point[0], wall_point[1], 'o', color='red')
            # plt.show()
            
            # add wall point
            points_sector = np.concatenate(([wall_point_inplane,], points_sector))
            mag_sector = np.concatenate(([0,], mag_sector))
            
            # plt.scatter(points_sector[:,0], points_sector[:,1], alpha=0.1)
            # plt.show()
        
            # Initial guesses for a, b, c, d
            initial_guess = [1, 1, 1, 1]
            result = least_squares(residuals_paraboloid, initial_guess, 
                                args=(points_sector[:,0], points_sector[:,1], mag_sector))
            # Extract the optimized parameters
            params_parabolid = result.x
            a_opt, b_opt, c_opt, d_opt = params_parabolid
            # print(f"Optimized parameters: a={a_opt}, b={b_opt}, c={c_opt}, d={d_opt}")

            # get gradient of wall point inplane and get WSS
            du_mag = derivate_magnitude(wall_point_inplane, params_parabolid)
            wss_i = du_mag*viscosity
            
            # save wall points and wss
            wss = np.append(wss,wss_i)
            wall_points = np.concatenate((wall_points, np.expand_dims(wall_point, axis=0)))
            # save over planes
            wss_plane.append(wss_i)
    
    # Compute WSS using 1d fit
    elif parabolic_sector_fitting == False:
        for idx, angle in enumerate(angles):     
            # get sector points
            # phi_center = angle/180*np.pi
        
            # maximum radius for sector of smoothed radius curve
            r_max = radii_smoothed[idx]   
            
            # # Select points exactly at phi_center (considering a small tolerance for precision)
            # tolerance = 2/180*np.pi  # Define a tolerance for floating-point comparisons
            # # phi_mask = np.abs(phi - phi_center) < tolerance
            # phi_mask = (phi >= phi_center-tolerance) & (phi <= phi_center+tolerance)
            # print(np.count_nonzero(phi_mask))
            
            # # Step 6: Create a mask for radial distance
            # r_mask = (r >= r_max-(r_layer+r_offset)) & (r <= r_max-r_offset)
            # z_mask = (translated_points[:, 2] >= z_plane - z_tol) & (translated_points[:, 2] <= z_plane + z_tol)
            # # Step 7: Combine masks to find points inside the sector
            # sector_mask = phi_mask & r_mask & z_mask
            
            # print(np.count_nonzero(sector_mask))
            
            wall_point = plane_origin + r_max * inplane_vectors[idx,:]
            # point1 = wall_point - inplane_vectors[idx,:] * 1*voxel_size
            # point2 = wall_point - inplane_vectors[idx,:] * 2*voxel_size
            # point3 = wall_point - inplane_vectors[idx,:] * 3*voxel_size
            # point4 = wall_point - inplane_vectors[idx,:] * 4*voxel_size
            
            # v_p1 = pv.PolyData(point1).sample(volume)
            # v_p2 = pv.PolyData(point2).sample(volume)
            # v_p3 = pv.PolyData(point3).sample(volume)
            # v_p4 = pv.PolyData(point4).sample(volume)

            # plt.plot(0, 0, 'o')
            # plt.plot(1, np.linalg.norm([v_p1['u'],v_p1['v'],v_p1['w']]), 'o')
            # plt.plot(2, np.linalg.norm([v_p2['u'],v_p2['v'],v_p2['w']]), 'o')
            # plt.plot(3, np.linalg.norm([v_p3['u'],v_p3['v'],v_p3['w']]), 'o')
            # plt.plot(4, np.linalg.norm([v_p4['u'],v_p4['v'],v_p4['w']]), 'o')
            
            plt.figure()
            plt.plot(0, 0, 'o', color='black')
            for i in range(1,20):
                point1 = wall_point - inplane_vectors[idx,:] * i*voxel_size
                
                v_p1 = pv.PolyData(point1).sample(volume)

                plt.plot(i*0.1, np.linalg.norm([v_p1['u'],v_p1['v'],v_p1['w']]), 'o')

            plt.show()
            
            1/0
            
            # get points in rotated volume
            points_sector = volume_rot.points[sector_mask]
            # get velocities
            u_sector = volume_rot['u'][sector_mask]
            v_sector = volume_rot['v'][sector_mask]
            w_sector = volume_rot['w'][sector_mask]
            vel_sector = np.vstack((u_sector, v_sector, w_sector)).T
            mag_sector = np.linalg.norm(vel_sector, axis=1)
            
            # get single wall point inplane of sector #r_max*inplane_vectors[idx] 
            wall_point_inplane = plane_origin + [r_max*np.cos(phi_center), r_max*np.sin(phi_center), 0]
            wall_point = plane_origin + r_max * inplane_vectors[idx] 

            # plt.scatter(points_sector[:,0], points_sector[:,1], alpha=0.1)
            # plt.plot(plane_origin[0], plane_origin[1], 'o', color='blue')
            # plt.plot(wall_point[0], wall_point[1], 'o', color='red')
            # plt.show()
            
            # add wall point
            points_sector = np.concatenate(([wall_point_inplane,], points_sector))
            mag_sector = np.concatenate(([0,], mag_sector))
            
            # plt.scatter(points_sector[:,0], points_sector[:,1], alpha=0.1)
            # plt.show()
        
            # Initial guesses for a, b, c, d
            initial_guess = [1, 1, 1, 1]
            result = least_squares(residuals_paraboloid, initial_guess, 
                                args=(points_sector[:,0], points_sector[:,1], mag_sector))
            # Extract the optimized parameters
            params_parabolid = result.x
            a_opt, b_opt, c_opt, d_opt = params_parabolid
            # print(f"Optimized parameters: a={a_opt}, b={b_opt}, c={c_opt}, d={d_opt}")

            # get gradient of wall point inplane and get WSS
            du_mag = derivate_magnitude(wall_point_inplane, params_parabolid)
            wss_i = du_mag*viscosity
            
            # save wall points and wss
            wss = np.append(wss,wss_i)
            wall_points = np.concatenate((wall_points, np.expand_dims(wall_point, axis=0)))
            # save over planes
            wss_plane.append(wss_i)
    
    # print("max wss ", np.max(wss_plane))
    # 1/0

    if plane_idx % 50 == 0:
        print("wss plane max   ", np.max(wss_plane))   
        print("wss plane mean  ", np.mean(wss_plane))   
        
    wss_planes[plane_idx,:] = wss_plane
    
    # plt.plot(wss_plane)
    # plt.title('WSS over cross section')
    # plt.show()
    
    t2 = time.time()
    if plane_idx % 10 == 0:
        print(f"Total time for {plane_idx} plane ", round(t2-t1,2), " seconds")

# ------------------------------------------------------------------
# wss evaluation
# ------------------------------------------------------------------
print("\nMean WSS    ", np.mean(wss))
print("Max WSS     ", np.max(wss))

# polydata object of wall points
wall_polydata = pv.PolyData(wall_points)
wall_polydata['wss'] = wss

plane_indices = np.repeat(np.arange(n_pts), len(angles))
wall_polydata['plane_idx'] = plane_indices

wall_polydata.save(target_path + f'/{wall_vtk_filename}.vtk')

# save surface points from raytracing
surface_polydata = pv.PolyData(np.reshape(surface_points, (n_pts*len(angles),3)))
surface_polydata.save(target_path + f'/{surface_vtk_filename}.vtk')

# wss[wss>500] = 500
# wall_polydata['wss'] = wss
# wall_polydata.save(target_path + f'/{wall_vtk_filename}_threshold.vtk')

# ------------------------------------------------------------------
# plot histogram
# ------------------------------------------------------------------
histogram, bin_edges = np.histogram(wss.ravel(), bins='auto', density=True)
cfd_histogram, cfd_bin_edges = np.histogram(cfd_wss, bins='auto', density=True)

plt.figure(figsize=(8, 6))
plt.plot(bin_edges[0:-1], histogram, alpha=0.5)  # Bin edges have one extra element
plt.plot(cfd_bin_edges[0:-1], cfd_histogram, alpha=0.5)  # Bin edges have one extra element
plt.xlabel(f'WSS [Pa]')
plt.ylabel('Count')
plt.grid(True)
plt.show()

histogram, bin_edges = np.histogram(wss.ravel(), bins='auto')#, density=True)
plt.figure(figsize=(8, 6))
plt.plot(bin_edges[0:-1], histogram, alpha=0.5)  # Bin edges have one extra element
plt.xlabel(f'WSS [Pa]')
plt.ylabel('Count')
plt.grid(True)
plt.show()


# end
1/0

# plot wall points
p = pv.Plotter(notebook=0)
p.add_mesh(wall_polydata, scalars='wss',  cmap='jet', point_size=2)
p.link_views()
p.show()

1/0
# generate surface
wall_surface = wall_polydata.delaunay_2d()
# wall_surface = wall_surface.extract_surface()
p = pv.Plotter(notebook=0, shape=(1,2))
p.add_mesh(wall_surface, scalars="wss", cmap="jet")
p.show()

1/0
p = pv.Plotter(notebook=0, shape=(1,2))
p.subplot(0, 0)
p.add_mesh(wall_polydata, show_edges=True, cmap='jet')
p.subplot(0, 1)
p.add_mesh(wall_surface, show_edges=True, cmap='jet')
p.link_views()
p.show()


1/0

# p = pv.Plotter(notebook=0)
# # p.add_mesh(plane, show_edges=True, cmap='jet')
# p.add_mesh(pc0)
# p.add_mesh(pc0_rot)
# glyphs = pc0.glyph(orient='Velocity', scale='Velocity', factor=0.5)
# p.add_mesh(glyphs, color='blue')  # Plot velocity glyphs
# p.show()

# p = pv.Plotter(notebook=0)
# # p.add_mesh(plane, show_edges=True, cmap='jet')
# # p.add_mesh(volume, show_edges=True, cmap='jet')
# p.add_mesh(pc0)
# # p.add_mesh(pc0_rot)
# p.add_mesh(pc1)
# # p.add_mesh(pc1_rot)
# p.add_mesh(pc2)
# # p.add_mesh(pc2_rot)
# p.add_mesh(pc3)
# p.show_axes() 
# p.show()

1/0

### plots
# p = pv.Plotter(notebook=0)
# # p.add_mesh(plane, show_edges=True, cmap='jet')
# p.add_mesh(plane, scalars='wall_mask', show_edges=True, cmap='jet')
# p.show()

# p = pv.Plotter(notebook=0, shape=(1,2))
# p.subplot(0, 0)
# p.add_mesh(surface, show_edges=True, cmap='jet')
# p.subplot(0, 1)
# # p.add_mesh(spline.points, show_edges=True, cmap='jet')
# p.add_mesh(np.gradient(spline.points,axis=1), show_edges=True, cmap='jet')
# p.link_views()
# p.show()

# p = pv.Plotter(notebook=0, shape=(1,2))
# p.subplot(0, 0)
# p.add_mesh(volume, show_edges=True, cmap='jet')
# p.subplot(0, 1)
# p.add_mesh(plane, scalars='mask', show_edges=True, cmap='jet')
# p.link_views()
# p.show()

1/0

# 10. Save the whole volume and the surface
# TODO: Improve this so we can save volume and surface in 1 vtk file
if save_to_vtk:
    volume.save("{}_volume.vtk".format(vtk_filename))
    surf.save("{}_surface.vtk".format(vtk_filename))

if show_plot:
    # ============ Plot the images ============ 
    p = pv.Plotter(notebook=0, shape=(2, 2), border=False)
    # p.set_background("lightgray") 
    
    # === Row 1, col 1
    scaled_volume = volume.copy()
    velocity_data = volume['Velocity']
    u_array, v_array, w_array = volume['u'], volume['v'], volume['w']
    velocity_data = velocity_data / 1000
    u_array = u_array / 1000
    v_array = v_array / 1000
    w_array = w_array / 1000
    scaled_volume["Velocity"] = velocity_data  
    scaled_volume['u'], scaled_volume['v'], scaled_volume['w'] = u_array, v_array, w_array 
    min_velocity = velocity_data.min()
    max_velocity = velocity_data.max()

    # print(min_velocity)
    # print(max_velocity)
    # print(max(volume['u']))
    # print(max(u_array))
    
    p.subplot(0, 0)
    p.add_text("Voxels\n{} mm".format(voxel_size), font_size=20, color='black')
    # p.add_mesh(volume, show_edges=True, cmap='jet')
    p.add_mesh(scaled_volume, scalars="Velocity", clim=[min_velocity, max_velocity], show_edges=True, cmap='jet')

    # === Row 1, col 2
    p.subplot(0, 1)
    p.add_text("Surface (n={})".format(smoothing_iteration), font_size=20, color='black')
    result = surf.sample(velocity)
    field_object = p.add_mesh(result, scalars="Velocity", opacity=0.7, cmap='viridis')
    # field_object.add_scalar_bar(label_font_size=10, title_font_size=12, color='black')
    # Add some arrows just because :)
    # arrows = surf.glyph(scale="Normals", orient="Normals", tolerance=0.1)
    arrows = surf.glyph(orient="Normals", tolerance=0.1)
    p.add_mesh(arrows, color="black")
    p.show_bounds(all_edges=True)

    # === Row 2, col 1
    p.subplot(1, 0)
    p.add_text("Wall points\n(inward {})".format(inward_distance), font_size=20, color='black')
    # Add the points only, for visualization purpose
    field_object = p.add_mesh(pc0, cmap='jet', opacity=0.5)
    # field_object.add_scalar_bar(label_font_size=10, title_font_size=12, color='black')
    # Add glyphs for pc1 and pc2
    pc1_arrows = pc1.glyph(orient='vectors', scale=False, factor=0.6,)
    pc2_arrows = pc2.glyph(orient='vectors', scale=False, factor=0.6,)
    p.add_mesh(pc1_arrows, color='black')
    p.add_mesh(pc2_arrows, color='red')

    # === Row 2, col 2
    p.subplot(1, 1)
    fitting_functext = "parabolic" if parabolic_fitting else "linear"
    # p.add_text("WSS ({})".format(fitting_functext), font_size=20, color='black')
    field_object = p.add_mesh(surf, scalars="wss", cmap='jet')
    # field_object.add_scalar_bar(label_font_size=10, title_font_size=12, color='black')

    if show_wss_contours:
        contours = surf.contour()
        p.add_mesh(contours, color='black', line_width=1)

    # ===================================
    # Link all the views and show the plot
    p.link_views()
    # p.show(full_screen=True)
    p.show()
