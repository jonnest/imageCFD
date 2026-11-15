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


# from plot_functions import *

target_path = '/Users/jonathan/Documents/applications/superres/ubend_1000_m1/vtk_eval'
# velocity_file = '../geom3/data_upsampled_r4_t0.h5'
# velocity_file = '../geom3/data_cubic_t0.h5'
# velocity_file = '../geom3/data_HR_r4_t0.h5'
# mask_file = '../geom3/data_HR_r4_t0.h5'

input_filepath = target_path + '/data_HR_r100.h5'
# input_filepath = target_path + '/data_LR_20_r200.h5'
voxel_size = 0.1  # in mm

# for 3d sector parabolid
r_offset = 1*voxel_size
r_layer = 3*voxel_size
# all as vector multiplication # 30*voxel_size = max sample radius
nr_radial = 50
# discretized ngles to analyze - multiple of 360 !
angle_step = 18

mask_filepath = input_filepath

plot_path = target_path + '/plots'
cfd_file = target_path + '/cfd/cfd_wss_1.vtk'

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
n_pts = 500
grid_voxel_size = 0.5 # 0.5

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

cfd_wss_max = np.linalg.norm(cfd_mesh.point_data['wallShearStress'],axis=1).max()*density
cfd_wss_mean = np.linalg.norm(cfd_mesh.point_data['wallShearStress'],axis=1).mean()*density

print("\nCFD WSS max [Pa]    ", cfd_wss_max)
print("CFD WSS mean [Pa]   ", cfd_wss_mean)


# ----------- Parameters ----------------------
# If the dimension is not isotropic, you might as well edit it here
spacing = (voxel_size,voxel_size,voxel_size) # in mm
inward_distance = inward_distance
smoothing_iteration = smoothing

# 1. Load velocity vectors and segmentation/mask
# Note: Please replace the loader depending on your data format, this is just an example
logger.info("Loading velocity vectors")
# u,v,w = load_vector_fields(input_filepath, ['u', 'v', 'w'], 0)
u,v,w = load_my_vector_fields(input_filepath, ['u', 'v', 'w'], 0)
# Convert velocities in mm/s
u,v,w = u*1000, v*1000, w*1000

logger.debug("Image shape {}".format(u.shape))
print("shape u   ", u.shape)
print("shape v   ", v.shape)
print("shape w   ", w.shape)
u,v,w = u.astype(np.float64), v.astype(np.float64), w.astype(np.float64)

logger.info("Load segmentation")
# mask = load_segmentation(mask_filepath, 'mask', 0)
mask = load_my_segmentation(mask_filepath, 'mask', 0)
mask = mask.astype(np.float64)

    
# 2. Construct uniform grid for vector values and mask
logger.info("Constructing uniform grids for vectors and mask")
velocity = create_uniform_vector(u,v,w, spacing)
mesh = create_uniform_grid(mask, spacing)

### TODO : why threshold # !!!!!!!!!!!!
# mesh = mesh.threshold_percent(40)
# mesh = mesh.threshold([0.3,0.8])

# 3. Get the volume within the mask (for visualization purpose)
logger.info("Get volume")
volume = mesh.sample(velocity)
volume.set_active_scalars("Velocity")

# 4. Get the surface of the mesh
# Note: do not use extract_geometry, the PyVista manual on the website is misleading
logger.info("Extracting surface")
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

### rename branches by order of outlet size
tf_branches, tf_gradients = reorderBranches(surface, tf_branches, tf_gradients)

# resample mesh
surface = uniformlyResampleMesh(surface,n_points=20000,n_subdivisions=2)

# window length for spline smoothing
window_length = int(to_closest_odd(0.2*n_pts))
smoothed_branch = smooth_3d_curve(tf_branches['0'], window_length, polyorder=2)

spline = pv.Spline(smoothed_branch, 1000)
spline.points = spline.points
spline.save(target_path + f'/smooth_spline.vtk')
            
# initialize wall points with wss
wall_points = np.empty((0,3))
wss = np.array([])
wss_planes = np.empty((0,0))


# loop over all center points
for plane_idx in range(n_pts):
# for plane_idx in range(10):
    t1 = time.time()
    # inflow 10/500
    # stenosis 150/500 or ?? 350/500
    # plane_idx = 150
    
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

    # number of interpolation points
    dr = voxel_size * np.arange(0,nr_radial)
    
    # initialize maximum radius array 
    max_radii = np.empty(len(inplane_vectors))
    
    ### 2 parameter optimization problem
    ### find largest radius with highest gradient
    
    ### are NOT in rotated coordinate system
    # create array of all considered plane points
    # inplane_pts = (angles, radials, 3)
    inplane_pts = np.empty((len(range(0, 360, angle_step)),nr_radial,3))
    for j, step in enumerate(dr):
        inplane_pts[:,j,:] = plane_origin + step*inplane_vectors
    # plot to verify
    # plt.scatter(inplane_pts[:,:,0].ravel(), inplane_pts[:,:,1].ravel())    
    # plt.show()
    # plt.plot(inplane_pts[:,:,2].ravel())
    # plt.title('control z axis coordinate')
    # plt.show()
    
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
    # init 
    wss_plane = []
    r_cs_idx = []
    r_cs = []
    r_cs_ = []
    # loop over all phi and radial direction vectors
    for idx, angle in enumerate(range(0, 360, angle_step)):
        if len(inplane_vectors) != len(range(0, 360, angle_step)):
            raise ValueError('Number of angles and vectors does not match')
        
        # Apply Gaussian filter for noise reduction
        sigma_mag = 2  # Smoothing parameter for Gaussian filter
        # filtered_mag = gaussian_filter1d(inplane_mag[idx], sigma_mag)
      
        # Optional: Use bilateral filtering for edge-preserving smoothing
        # (Requires scaling data between 0 and 1 if using skimage)
        # filtered_mag = denoise_bilateral(inplane_mag[idx], sigma_color=0.1, sigma_spatial=0.1)
        filtered_mag = inplane_mag[idx, ...]
        
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
        
        max_radii[idx] = max_grad_dr_
        
        ### plot spline
        # plt.plot(dr, inplane_mag[idx,:], 'b')
        # plt.plot(dr_, spline(dr_), 'r')
        # plt.plot(max_grad_dr_, 0, 'o')
        # plt.show()
        
        # get max_grad index of coarse dr
        r = np.argmin(np.abs(dr - max_grad_dr_))
        r_cs_idx.append(r)
        r_cs_.append(max_grad_dr_)
        r_cs.append(dr[r])
    
        
    # smooth 
    sigma = np.sqrt(np.var(max_radii))*10
    # extend array and make periodic
    padding = len(max_radii)
    extended_radii = np.concatenate([max_radii[-padding:], max_radii, max_radii[:padding]])
    extended_smoothed_radii = gaussian_filter1d(extended_radii, sigma=sigma)  
    smoothed_radii = extended_smoothed_radii[padding:-padding]
    
    # plt.plot(max_radii)
    # plt.plot(smoothed_radii)
    # plt.show()
    
    if plane_idx % 50 == 0:
        plt.plot(max_radii)
        plt.plot(smoothed_radii)
        # plt.show()
        plt.savefig(plot_path + '/' + f'radii_plane{plane_idx}.png', dpi=200, bbox_inches='tight')
    
        
    # delta phi of sector - # default 20
    phi_sector = angle_step*1.5
    for idx, angle in enumerate(range(0, 360, angle_step)):
        
        # get sector points
        phi_center = angle/180*np.pi
        delta_phi = phi_sector/180*np.pi
        z_tol = voxel_size/2
        
        # maximum radius for sector
        r_max = smoothed_radii[idx]
        
        translated_points = volume_rot.points - plane_origin
        translated_inplane_pts = inplane_pts_pv.points - plane_origin
        z_plane = np.mean(translated_inplane_pts[...,2])
    
        r = np.linalg.norm(translated_points[:, :2], axis=1)  # Radial distances (x, y components only)
        phi = np.arctan2(translated_points[:, 1], translated_points[:, 0])  # Angles in radians

        phi = np.mod(phi, 2 * np.pi)
        
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
        r_mask = (r >= r_max-(r_layer+r_offset)) & (r <= r_max-r_offset)

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
        wall_point = plane_origin + [r_max*np.cos(phi_center), r_max*np.sin(phi_center), 0]
        
        # plt.scatter(points_sector[:,0], points_sector[:,1], alpha=0.1)
        # plt.plot(plane_origin[0], plane_origin[1], 'o', color='blue')
        # plt.plot(wall_point[0], wall_point[1], 'o', color='red')
        # plt.show()
        
        # add wall point
        points_sector = np.concatenate(([wall_point,], points_sector))
        mag_sector = np.concatenate(([0,], mag_sector))
        
        # plt.scatter(points_sector[:,0], points_sector[:,1], alpha=0.1)
        # plt.show()
    
        # Initial guesses for a, b, c, d
        initial_guess = [1, 1, 1, 1]
        result = least_squares(residuals, initial_guess, 
                               args=(points_sector[:,0], points_sector[:,1], mag_sector))
        # Extract the optimized parameters
        params_parabolid = result.x
        a_opt, b_opt, c_opt, d_opt = params_parabolid
        # print(f"Optimized parameters: a={a_opt}, b={b_opt}, c={c_opt}, d={d_opt}")

        # get gradient on wall and get WSS
        du_mag = derivate_magnitude(wall_point, params_parabolid)
        wss_i = du_mag*viscosity
        
        # save wall points and wss
        wss = np.append(wss,wss_i)
        wall_points = np.concatenate((wall_points, np.expand_dims(wall_point, axis=0)))
        # save over planes
        wss_plane.append(wss_i)
        
        # # plot sector
        # fig = plt.figure()
        # # ax = fig.add_subplot(111, projection='3d')
        # # ax = ax.plot_surface(inplane_pts_array[:,0], inplane_pts_array[:,1], radial_mag)
        # # surface = ax.plot_trisurf(parab_pts[:,0], parab_pts[:,1], parab_mag)
        # # surface = ax.scatter(parab_pts[:,0], parab_pts[:,1])
        # plt.scatter(parab_pts[:,0], parab_pts[:,1])
        # plt.show()
        
    wss_planes = np.concatenate((wss, wss_plane))
    
    print(f"Wss plane {plane_idx} max   ", np.max(wss_planes))   
    print(f"Wss plane {plane_idx} mean  ", np.mean(wss_planes))   

    # plt.scatter(parab_pts[:,0], parab_pts[:,1])
    # plt.title('parabola points in x,y plane')
    # plt.show()

    # plt.plot(r_cs)
    # plt.title('radius over cross section')
    # plt.show()
    
    # plt.plot(wss_plane)
    # plt.title('WSS over cross section')
    # plt.show()
    
    t2 = time.time()
    if plane_idx % 10 == 0:
        print(f"Total time for {plane_idx} plane ", round(t2-t1,2), " seconds")

# wss
print("\nMean WSS    ", np.mean(wss))
print("Max WSS     ", np.max(wss))

# polydata object of wall points
wall_polydata = pv.PolyData(wall_points)
wall_polydata['wss'] = wss
wss[wss>500] = 500
wall_polydata['wss'] = wss

# plot histogram
histogram, bin_edges = np.histogram(wss.ravel(), bins='auto')
plt.figure(figsize=(8, 6))
plt.plot(bin_edges[0:-1], histogram)  # Bin edges have one extra element
plt.xlabel(f'WSS [Pa]')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# end
1/0

# plot wall points
p = pv.Plotter(notebook=0)
p.add_mesh(wall_polydata, scalars='wss',  cmap='jet', point_size=10)
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
    logger.info("Saving to {}_surface.vtk".format(vtk_filename))
    volume.save("{}_volume.vtk".format(vtk_filename))
    surf.save("{}_surface.vtk".format(vtk_filename))

if show_plot:
    # ============ Plot the images ============ 
    logger.info("Preparing plot...")
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
