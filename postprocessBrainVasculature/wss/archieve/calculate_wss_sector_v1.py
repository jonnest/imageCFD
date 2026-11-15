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

from wss_utils import *
from surfaceMesh import *
from gridOperators import *
from scipy.ndimage import binary_erosion
from scipy.optimize import least_squares
from scipy.interpolate import griddata


target_path = '/Users/jonathan/Documents/applications/superres/ubend_1000_m1/vtk_eval'
# velocity_file = '../geom3/data_upsampled_r4_t0.h5'
# velocity_file = '../geom3/data_cubic_t0.h5'
# velocity_file = '../geom3/data_HR_r4_t0.h5'
# mask_file = '../geom3/data_HR_r4_t0.h5'

input_filepath = target_path + '/data_HR_r100.h5'
# input_filepath = target_path + '/data_LR_20_r200.h5'
voxel_size = 0.1  # in mm
wall_vtk_filename = 'wall_point_HR_r100'

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
# Number of iterations to adjust point coordinates using Laplacian smoothing
smoothing = 500 
parabolic_fitting = True
# no slip condition
no_slip_condition = True
# Show WSS contours for visualization
show_wss_contours = True
# Save volume mesh and surface WSS in vtk file
save_to_vtk = True
# Prefix for the output file when save_to_vtk is True
vtk_filename = 'result'
# Logging level
loglevel = logging.INFO

# centerline extraction by voxelization
n_pts = 100
grid_voxel_size = 0.5 # 0.5

# ----------- Parameters ----------------------
# If the dimension is not isotropic, you might as well edit it here
spacing = (voxel_size,voxel_size,voxel_size) # in mm
inward_distance = inward_distance
smoothing_iteration = smoothing

# 1. Load velocity vectors and segmentation/mask
# Note: Please replace the loader depending on your data format, this is just an example
# u,v,w = load_vector_fields(input_filepath, ['u', 'v', 'w'], 0)
u,v,w = load_my_vector_fields(input_filepath, ['u', 'v', 'w'], 0)
# Convert velocities in mm/s
u,v,w = u*1000, v*1000, w*1000

print("shape u   ", u.shape)
print("shape v   ", v.shape)
print("shape w   ", w.shape)
u,v,w = u.astype(np.float64), v.astype(np.float64), w.astype(np.float64)

# mask = load_segmentation(mask_filepath, 'mask', 0)
mask = load_my_segmentation(mask_filepath, 'mask', 0)
mask = mask.astype(np.float64)
print(mask.dtype)
    
# 2. Construct uniform grid for vector values and mask
velocity = create_uniform_vector(u,v,w, spacing)
mesh = create_uniform_grid(mask, spacing)
masked_mesh = mesh.threshold(scalars="mask", value=0.5)  # Adjust threshold as needed
surf = masked_mesh.extract_surface()
# surf = mesh.extract_surface()
surf = surf.smooth(n_iter=smoothing_iteration)

# 3. Get the volume within the mask (for visualization purpose)
volume = mesh.sample(velocity)
volume.set_active_scalars("Velocity")

# 4. Get the surface of the mesh
# Note: do not use extract_geometry, the PyVista manual on the website is misleading
surf = surf.smooth(n_iter=smoothing_iteration)

# get cross section points
surface = surf.triangulate()
surface = pymeshfix.MeshFix(surface)
surface.repair(joincomp=True, remove_smallest_components=True) 
surface = surface.mesh        
print(f"The surface still has {surface.n_open_edges} open edges.")

grid_voxel_size = 0.5 # 0.5
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
# TODO : smoothed gradient - how it behaves
smoothed_gradient = smooth_3d_curve(tf_gradients['0'], window_length, polyorder=2)

# set angles to compute wss on
angles = range(0, 360, angle_step)

# initialize wall points with wss
wall_points = np.empty((0,3))
wss = np.empty((0))

# loop over all center points
for plane_idx in range(n_pts):
# for plane_idx in range(10):
    
    t1 = time.time()

    # boundaries = extractEdges_fromBranches(surface, tf_branches['0'], tf_gradients['0'], target_index=plane_idx, path=target_path)
    boundaries = extractEdges_fromBranches(surface, smoothed_branch, smoothed_gradient, target_index=plane_idx, path=target_path)

    # center point and radii of cross section
    # ctr_point = tf_branches['0'][plane_idx]
    ctr_point = smoothed_branch[plane_idx]

    radii = np.linalg.norm(ctr_point - boundaries.points, axis=1)

    # cross_section, plane_normal, plane_ctr = get_cross_section(volume, tf_branches['0'], plane_idx, voxel_size, radii)
    cross_section, plane_normal, plane_ctr = get_cross_section(volume, smoothed_branch, plane_idx, voxel_size, radii)

    plane = cross_section.sample(volume)

    plane_mask = np.ones(cross_section.points.shape)
    cross_section['plane_mask'] = plane_mask
    surf_masked = surf.sample(cross_section)# , categorical=True)

    sliced_surface = surface.slice(normal=plane_normal, origin=plane_ctr)

    # Compute distances and find the closest surface points
    # TODO : problem - slicing cuts through whole geometry, can cut multiple times
    
    intersection_points = sliced_surface.points
    surface_points = surface.points
    closest_points = []
    for ipoint in intersection_points:
        distances = np.linalg.norm(surface_points - ipoint, axis=1)
        closest_point_index = np.argmin(distances)
        closest_points.append(surface_points[closest_point_index])

    # Convert closest points to PyVista format for visualization
    closest_points = np.array(closest_points)
    closest_points_cloud = pv.PolyData(closest_points)

    # Step 5: Visualization
    # p = pv.Plotter()
    # # p.add_mesh(cross_section, color='lightblue', opacity=0.5, label='Surface')
    # # p.add_mesh(sliced_surface, color='red', line_width=3, label='Intersection')
    # # p.add_mesh(closest_points_cloud, color='green', point_size=10, render_points_as_spheres=True, label='Closest Points')
    # p.add_mesh(surface.clip_surface(plane), color='lightblue', opacity=0.5, label='Surface')
    # p.add_legend()
    # p.show()


    # p = pv.Plotter(notebook=0)
    # p.add_mesh(cross_section, show_edges=True, cmap='jet') #, opacity=0.3)
    # # p.add_mesh(sliced_surface, show_edges=True, cmap='jet') #, opacity=0.3)
    # p.add_mesh(surf_masked, scalars='plane_mask', show_edges=True, cmap='jet') #, opacity=0.3)
    # p.show_axes() 
    # p.show_grid()
    # p.show()
    
    # p = pv.Plotter(notebook=0)
    # # p.add_mesh(volume, show_edges=True, cmap='jet')
    # p.add_mesh(plane, show_edges=True, cmap='jet')
    # p.add_mesh(boundaries, show_edges=True, cmap='jet')
    # p.add_mesh(pv.PolyData(ctr_point), show_edges=True, cmap='jet')
    # p.show_axes() 
    # p.show_grid()
    # p.show()

    # CELL get normal vectors of surface
    surf.compute_normals(inplace=True)

    # POINT normal vectors on surface 
    surf_normals_pts = surf.point_normals

    # # sample boundary points, including normal vector
    # pc0_plane = plane.sample(surf)

    # get points on wall
    pc0 = plane.points[plane['wall_mask'].astype(bool)]
    pc0 = order_points_circle(pc0)
    
    if plane_idx==0:
        print("pc0 nr points ", pc0.shape)
    
    # 6. Compute the coordinates from surface (pc0) to the equidistant points in the inward normal (pc1 and pc2)
    pc0 = pv.PolyData(pc0)
    pc0 = pc0.sample(velocity)
    pc0.set_active_scalars("Velocity")

    '''
    p = pv.Plotter(notebook=0)
    p.add_mesh(plane, scalars='wall_mask', show_edges=True, cmap='jet')
    p.add_mesh(pc0, show_edges=True, cmap='jet')
    p.add_mesh(pv.PolyData(ctr_point), show_edges=True, cmap='jet')
    p.show_axes() 
    p.show_grid()
    p.show()
    '''

    # get normals of point data - scipy interpolation
    # pc0_surf_normals = griddata(surf.points, surf_normals_pts, pc0.points, method='linear')
    pc0_surf_normals = griddata(surf.points, surf_normals_pts, pc0.points, method='nearest')
    if np.isnan(pc0_surf_normals).any()==True:
        print("NaN values in pc0_surf_normals")
            
    # Construct the first inward point clouds form surface (pc1) and get the velocity values
    loc1 = pc0.points - (inward_distance * pc0_surf_normals)
    pc1 = pv.PolyData(loc1)
    pc1 = pc1.sample(velocity) # sample velocity on points

    # Construct the second inward point clouds form surface (pc2) and get the velocity values
    loc2 = pc0.points - (2 * inward_distance * pc0_surf_normals)
    pc2 = pv.PolyData(loc2)
    pc2 = pc2.sample(velocity) # sample velocity on points

    # Construct the second inward point clouds form surface (pc2) and get the velocity values
    loc3 = pc0.points - (3 * inward_distance * pc0_surf_normals)
    pc3 = pv.PolyData(loc3)
    pc3 = pc3.sample(velocity) # sample velocity on points

    ### rotate points into plane
    # TODO : rotation of scalar values doesn't change values
    # if there is a velocity vector, then rotation will change vector?
    pc0_vel = np.vstack((pc0['u'], pc0['v'], pc0['w'])).T
    pc1_vel = np.vstack((pc1['u'], pc1['v'], pc1['w'])).T
    pc2_vel = np.vstack((pc2['u'], pc2['v'], pc2['w'])).T
    pc3_vel = np.vstack((pc3['u'], pc3['v'], pc3['w'])).T

    pc0['u_vec'] = pc0_vel
    pc1['u_vec'] = pc1_vel
    pc2['u_vec'] = pc2_vel
    pc3['u_vec'] = pc3_vel

    pc0_rot = rotate_pointcloud_to_plane(pc0, plane_normal, [0,0,1], plane_ctr)
    pc1_rot = rotate_pointcloud_to_plane(pc1, plane_normal, [0,0,1], plane_ctr)
    pc2_rot = rotate_pointcloud_to_plane(pc2, plane_normal, [0,0,1], plane_ctr)
    pc3_rot = rotate_pointcloud_to_plane(pc3, plane_normal, [0,0,1], plane_ctr)
    # compute velocity magnitudes from velocity vector safed as points
    pc0_mag_rot = np.linalg.norm(pc0_rot['u_vec'], axis=1)
    pc1_mag_rot = np.linalg.norm(pc1_rot['u_vec'], axis=1)
    pc2_mag_rot = np.linalg.norm(pc2_rot['u_vec'], axis=1)
    pc3_mag_rot = np.linalg.norm(pc3_rot['u_vec'], axis=1)

    
    # # get velocity vector for all rings
    # pc0_vel = np.vstack((pc0['u'], pc0['v'], pc0['w'])).T
    # pc1_vel = np.vstack((pc1['u'], pc1['v'], pc1['w'])).T
    # pc2_vel = np.vstack((pc2['u'], pc2['v'], pc2['w'])).T
    # pc3_vel = np.vstack((pc3['u'], pc3['v'], pc3['w'])).T
    # # rotate velocity vectors as point vectors, velocity vector is saved in pcX_vel_rot.points
    # pc0_vel_rot = rotate_pointcloud_to_plane(pc0_vel, plane_normal, [0,0,1], plane_ctr)
    # pc1_vel_rot = rotate_pointcloud_to_plane(pc1_vel, plane_normal, [0,0,1], plane_ctr)
    # pc2_vel_rot = rotate_pointcloud_to_plane(pc2_vel, plane_normal, [0,0,1], plane_ctr)
    # pc3_vel_rot = rotate_pointcloud_to_plane(pc3_vel, plane_normal, [0,0,1], plane_ctr)
    # # compute velocity magnitudes from velocity vector safed as points
    # pc0_mag_rot = np.linalg.norm(pc0_vel_rot.points, axis=1)
    # pc1_mag_rot = np.linalg.norm(pc1_vel_rot.points, axis=1)
    # pc2_mag_rot = np.linalg.norm(pc2_vel_rot.points, axis=1)
    # pc3_mag_rot = np.linalg.norm(pc3_vel_rot.points, axis=1)

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

    # number of points in circumference direction
    d_phi = 2
    wss_plane = []

    for p in range(pc0_rot.points.shape[0]):
        # least square for sector parabolid in x,y plane
        # MAGNITUDE as axial velocity
        
        # periodic selection of points
        N = len(pc0_rot.points)
        x_sector = np.concatenate(([pc0_rot.points[p,  0],], # only middle point of sector on wall
                                    pc1_rot.points[[(p + i) % N for i in range(-d_phi, d_phi+1)],0],
                                    pc2_rot.points[[(p + i) % N for i in range(-d_phi, d_phi+1)],0],
                                    pc3_rot.points[[(p + i) % N for i in range(-d_phi, d_phi+1)],0] ))

        y_sector = np.concatenate(([pc0_rot.points[p,  1],], # only middle point of sector on wall
                                    pc1_rot.points[[(p + i) % N for i in range(-d_phi, d_phi+1)],1],
                                    pc2_rot.points[[(p + i) % N for i in range(-d_phi, d_phi+1)],1],
                                    pc3_rot.points[[(p + i) % N for i in range(-d_phi, d_phi+1)],1] ))

        vel_mag_obs = np.concatenate(([0,],                # wall velocity zero 
                                    pc1_mag_rot[[(p + i) % N for i in range(-d_phi, d_phi+1)]],
                                    pc2_mag_rot[[(p + i) % N for i in range(-d_phi, d_phi+1)]],
                                    pc3_mag_rot[[(p + i) % N for i in range(-d_phi, d_phi+1)]] ))

        # if p==0:
        #     print("Number of points used for parabolid sector ", len(x_sector))
            
        # Initial guesses for a, b, c, d
        initial_guess = [1, 1, 1, 1]
        result = least_squares(residuals_paraboloid, initial_guess, args=(x_sector, y_sector, vel_mag_obs))
        # Extract the optimized parameters
        params_parabolid = result.x
        a_opt, b_opt, c_opt, d_opt = params_parabolid
        # print(f"Optimized parameters: a={a_opt}, b={b_opt}, c={c_opt}, d={d_opt}")

        # get gradient on wall and get WSS
        du_mag = derivate_magnitude(pc2_rot.points[2, :], params_parabolid)
        wss_i = du_mag*viscosity
        
        wss_plane.append(wss_i)

    # save wall points and wss
    wall_points = np.concatenate((wall_points, pc0.points))
    wss = np.concatenate((wss, wss_plane))
    
    t2 = time.time()
    if plane_idx % 10 == 0:
        print(f"Total time for {plane_idx} plane ", round(t2-t1,2), " seconds")

# wss
print("\nMean WSS    ", np.mean(wss))
print("Max WSS     ", np.max(wss))

# polydata object of wall points
wall_polydata = pv.PolyData(wall_points)
wall_polydata['wss'] = wss

plane_indices = np.repeat(np.arange(n_pts), len(angles))
wall_polydata['plane_idx'] = plane_indices

wall_polydata.save(target_path + f'/{wall_vtk_filename}.vtk')



1/0
# plot wall points
p = pv.Plotter(notebook=0)
p.add_mesh(wall_polydata, scalars='wss',  cmap='plasma')
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


####################################################################################
####################################################################################
####################################################################################

# 5. Calculate the point normals (depends on what we computed on cells or on nodes)
# Make sure we compute only the point normals, not the cell normals, they have the same array name
surf.compute_normals(point_normals=True, cell_normals=True, inplace=True, flip_normals=True)


# 6. Compute the coordinates from surface (pc0) to the equidistant points in the inward normal (pc1 and pc2)
# Construct a polydata of points from the surface points
pc0 = pv.PolyData(surf.points)
pc0 = pc0.sample(velocity)
pc0.set_active_scalars("Velocity")

# Construct the first inward point clouds form surface (pc1) and get the velocity values
loc1 = pc0.points + (inward_distance * surf.point_normals)
pc1 = pv.PolyData(loc1)
pc1 = pc1.sample(velocity)

# Construct the second inward point clouds form surface (pc2) and get the velocity values
loc2 = pc0.points + (2 * inward_distance * surf.point_normals)
pc2 = pv.PolyData(loc2)
pc2 = pc2.sample(velocity)

# 7. Calculate the tangential vector on the wall and inward points
if no_slip_condition:
    # For the velocity on the wall to zero
    pc0_tangent_mag = np.zeros(len(pc0.points))
else:
    # Extract the velocity on the wall. This might not be accurate as it depends on the segmentation
    pc0_vectors = extract_vectors(pc0)
    pc0_normals, pc0_tangent = get_orthogonal_vectors(pc0_vectors, surf.point_normals)
    pc0_tangent_mag = get_vector_magnitude(pc0_tangent)

# get pc1 vector normal and tangential
pc1_vectors = extract_vectors(pc1)
pc1_normals, pc1_tangent = get_orthogonal_vectors(pc1_vectors, surf.point_normals)
pc1_tangent_mag = get_vector_magnitude(pc1_tangent)
pc1['vectors'] = pc1_tangent

# get pc2 vector normal and tangential
pc2_vectors = extract_vectors(pc2)
pc2_normals, pc2_tangent = get_orthogonal_vectors(pc2_vectors, surf.point_normals)
pc2_tangent_mag = get_vector_magnitude(pc2_tangent)
pc2['vectors'] = pc2_tangent

# === Experimental ==
# Check if the tangential vectors of pc1 and pc2 are in the same direction 
c = pc1_tangent * pc2_tangent 
c = np.sum(c, axis=1)
c = c.clip(min = -1, max = 1) # Assign the direction
pc2_tangent_mag =  c * pc2_tangent_mag # Give negative magnitude value if pc2 is in different direction
# We don't check based on pc0 because we cannot trust the direction of pc0 vectors which may be affected by segmentation
## == end ==

# 8. Compute the gradients for every points
gradients = calculate_gradient(pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag, inward_distance, use_parabolic=parabolic_fitting)

# 9. Assign the gradients back to the surface
# gradients in 1/s
wss = gradients * viscosity

surf["wss"] = wss
# use the pc1 vectors for visualization
surf['wss_vectors'] = pc1_tangent

print("\nMean WSS [Pa]    ", round(np.mean(wss),8))
print("Max WSS  [Pa]    ", round(np.max(wss),8), "\n")

# plot histogram
histogram, bin_edges = np.histogram(wss.ravel(), bins='auto')
plt.figure(figsize=(8, 6))
plt.plot(bin_edges[0:-1], histogram)  # Bin edges have one extra element
# plt.title('Histogram')
plt.xlabel(f'WSS [Pa]')
plt.ylabel('Count')
plt.grid(True)
plt.show()

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
