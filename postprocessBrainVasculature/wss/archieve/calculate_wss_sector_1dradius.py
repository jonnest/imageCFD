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

from wss_utils import *
from surfaceMesh import *
from gridOperators import *
from scipy.ndimage import binary_erosion
from scipy.optimize import least_squares
from scipy.interpolate import griddata, UnivariateSpline
import pymeshfix

from mpl_toolkits.mplot3d import Axes3D

# from plot_functions import *

target_path = '/Users/jonathan/Documents/applications/superres/ubend_1000_m1/vtk_eval'
# velocity_file = '../geom3/data_upsampled_r4_t0.h5'
# velocity_file = '../geom3/data_cubic_t0.h5'
# velocity_file = '../geom3/data_HR_r4_t0.h5'
# mask_file = '../geom3/data_HR_r4_t0.h5'

# velocity_file = '../ubend_v1/data_LR_20.h5'
input_filepath = target_path + '/data_HR_r100.h5'
mask_filepath = input_filepath

plot_path = target_path + '/plots'

cfd_file = target_path + '/cfd/cfd_wss_1.vtk'

# input parameters
voxel_size = 0.1  # in mm
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

print("Original cfd point ", cfd_point_stenosis)
# read CFD
cfd_mesh = pv.read(cfd_file)


# translate both meshes to same origin and scaling
cfd_mesh = cfd_mesh.scale(1000)
new_origin = (0,0,0)
translation_cfd = tuple(new_origin[i] - cfd_mesh.center[i] for i in range(3))
# translate mesh and point
cfd_mesh.translate(translation_cfd, inplace=True)
pv.PolyData(cfd_point_stenosis).translate(translation_cfd, inplace=True)
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
print(mask.dtype)
    
# 2. Construct uniform grid for vector values and mask
logger.info("Constructing uniform grids for vectors and mask")
velocity = create_uniform_vector(u,v,w, spacing)
mesh = create_uniform_grid(mask, spacing)
mesh = mesh.threshold_percent(40)
# mesh = mesh.threshold([0.3,0.8])

# 3. Get the volume within the mask (for visualization purpose)
logger.info("Get volume")
volume = mesh.sample(velocity)
volume.set_active_scalars("Velocity")

# 4. Get the surface of the mesh
# Note: do not use extract_geometry, the PyVista manual on the website is misleading
logger.info("Extracting surface")
surf = mesh.extract_surface()
surf = surf.smooth(n_iter=smoothing_iteration)

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
# TODO : smoothed gradient - how it behaves
smoothed_gradient = smooth_3d_curve(tf_gradients['0'], window_length, polyorder=2)


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
    plane_idx = 250
    
    ### analysis in rotated reference coordinate system and polar coordinates
    # get normal vector of current plane with point origin 
    # smoothed_gradient or tf_branches['0'], tf_gradients['0'],
    plane_normal = smoothed_gradient[plane_idx]/np.linalg.norm(smoothed_gradient[plane_idx])
    plane_origin = smoothed_branch[plane_idx]
    
    # rotation all points for x,y plane and normal is z-axis
    euler_rotations = align_vectors(plane_normal, [0,0,1])
    volume_rot = volume.copy()
    volume_rot = volume_rot.rotate_x(euler_rotations[0],point=plane_origin, transform_all_input_vectors=True)
    volume_rot = volume_rot.rotate_y(euler_rotations[1],point=plane_origin, transform_all_input_vectors=True)
    volume_rot = volume_rot.rotate_z(euler_rotations[2],point=plane_origin, transform_all_input_vectors=True)
    
    
    # loop over phi and get velocity curve in each radial direction
    angle_step = 5 # multiple of 360 !
    inplane_vectors = generate_vectors_in_plane(plane_normal, angle_step=angle_step)

    # number of interpolation points
    # TODO : all as vector multiplication
    nr_radial = 30 # 30*voxel_size = max sample radius
    dr = voxel_size * np.arange(0,nr_radial)
    
    # initialize maximum radius array 
    max_radii = np.empty(len(inplane_vectors))
    
    ### 2 parameter optimization problem
    ### find largest radius with highest gradient
    # create array of all considered plane points
    # inplane_pts = (angles, radials, 3)
    inplane_pts = np.empty((len(range(0, 360, angle_step)),nr_radial,3))
    for j, step in enumerate(dr):
        inplane_pts[:,j,:] = plane_origin + step*inplane_vectors
    # plot to verify
    # plt.scatter(inplane_pts[:,:,0].ravel(), inplane_pts[:,:,1].ravel())    
    # plt.plot(inplane_pts[:,:,2].ravel())
    # plt.title('control z axis coordinate')
    # plt.show()
    
    # sample volume velocities on points of intererst
    inplane_pts_array = np.reshape(inplane_pts, (-1,3))
    radial_pts_pv = pv.PolyData(inplane_pts_array).sample(volume)
    # NOTE : when computing the magnitude, the sign is lost
    inplane_vel_array = np.vstack((radial_pts_pv['u'], radial_pts_pv['v'], radial_pts_pv['w'])).T
    inplane_mag_array = np.linalg.norm(inplane_vel_array, axis=1)      
    # reshape back to 2d array to access via phi and dr coordinates
    inplane_mag = np.reshape(inplane_mag_array, inplane_pts.shape[:-1])
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax = ax.plot_surface(inplane_pts_array[:,0], inplane_pts_array[:,1], radial_mag)
    # surface = ax.plot_trisurf(inplane_pts_array[:,0], inplane_pts_array[:,1], inplane_mag_array)
    # plt.show()
    
    # init 
    wss_plane = []
    r_cs_idx = []
    r_cs = []
    # loop over all phi and radial direction vectors
    for idx, angle in enumerate(range(0, 360, angle_step)):
        if len(inplane_vectors) != len(range(0, 360, angle_step)):
            raise ValueError('Number of angles and vectors does not match')
        
        ### find largest gradient 
        ### TODO : alternative - iterate from center out to find highest grad
        # fit univariate spline
        spline = UnivariateSpline(dr, inplane_mag[idx,:], s=4000)
        dx_spline = spline.derivative()
        
        # high res evaluation of spline
        dr_ = voxel_size * np.arange(0, nr_radial, 0.01)
        spline_grads = dx_spline(dr_)
        
        # find max gradient idx
        max_grad_idx_ = np.argmax(np.abs(spline_grads))
        max_grad_dr_ = dr_[max_grad_idx_]
        max_grad_ = spline_grads[max_grad_idx_]
        
        max_radii[idx] = max_grad_dr_
        
        plt.plot(dr, inplane_mag[idx,:], 'b')
        plt.plot(dr_, spline(dr_), 'r')
        plt.plot(max_grad_dr_, 0, 'o')
        plt.show()
        
        # get max_grad index of coarse dr
        r = np.argmin(np.abs(dr - max_grad_dr_))
        r_cs_idx.append(r)
        r_cs.append(max_grad_dr_)
        
        # sectored 3d parabolid (sp)
        # get indices of sector with one wall point 
        dphi = 5
        nr_dr = 6
        n_phi = len(range(0,360,angle_step))
        
        parab_pts = np.array([inplane_pts[idx, r,  :],])
        for dr_i in range(1,nr_dr+1):
            parab_pts = np.concatenate((parab_pts, inplane_pts[[(idx + i) % n_phi for i in range(-dphi, dphi+1)], r-dr_i, :]))

        parab_mag = np.array([0,])
        for dr_i in range(1,nr_dr+1):
            parab_mag = np.concatenate((parab_mag, inplane_mag[[(idx + i) % n_phi for i in range(-dphi, dphi+1)], r-1]))
        
        x_sector = parab_pts[:,0]
        y_sector = parab_pts[:,1]

        # Initial guesses for a, b, c, d
        initial_guess = [1, 1, 1, 1]
        result = least_squares(residuals_paraboloid, initial_guess, args=(x_sector, y_sector, parab_mag))
        # Extract the optimized parameters
        params_parabolid = result.x
        a_opt, b_opt, c_opt, d_opt = params_parabolid
        # print(f"Optimized parameters: a={a_opt}, b={b_opt}, c={c_opt}, d={d_opt}")

        # get gradient on wall and get WSS
        du_mag = derivate_magnitude(inplane_pts[idx, r,  :], params_parabolid)
        wss_i = du_mag*viscosity
        
        # save wall points and wss
        wss = np.append(wss,wss_i)
        wall_points = np.concatenate((wall_points, np.expand_dims(inplane_pts[idx, r, :],axis=0)))
        # save over planes
        wss_plane.append(wss_i)
        
        # # plot sector
        # fig = plt.figure()
        # # ax = fig.add_subplot(111, projection='3d')
        # # ax = ax.plot_surface(inplane_pts_array[:,0], inplane_pts_array[:,1], radial_mag)
        # # surface = ax.plot_trisurf(parab_pts[:,0], parab_pts[:,1], parab_mag)
        # # surface = ax.scatter(parab_pts[:,0], parab_pts[:,1])
        plt.scatter(parab_pts[:,0], parab_pts[:,1])
        plt.show()
        
        1/0
        
    wss_planes = np.concatenate((wss, wss_plane))
            
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
# wss[wss>500] = 500
# wall_polydata['wss'] = wss

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


####################################################################################
####################################################################################
####################################################################################

# 5. Calculate the point normals (depends on what we computed on cells or on nodes)
# Make sure we compute only the point normals, not the cell normals, they have the same array name
logger.info("Computing surface normals")
surf.compute_normals(point_normals=True, cell_normals=True, inplace=True, flip_normals=True)
logger.debug("{} {} normal points".format(surf.point_normals[0:2], len(surf.point_normals)))
logger.debug("{} {} points coords".format(surf.points[0:2], len(surf.points)))

# 6. Compute the coordinates from surface (pc0) to the equidistant points in the inward normal (pc1 and pc2)
# Construct a polydata of points from the surface points
logger.info("Probing velocity from surface and equidistant points {} nodes".format(len(surf.points)))
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
logger.info("Calculate normal and tangential velocity vectors")
if no_slip_condition:
    # For the velocity on the wall to zero
    pc0_tangent_mag = np.zeros(len(pc0.points))
else:
    # Extract the velocity on the wall. This might not be accurate as it depends on the segmentation
    pc0_vectors = extract_vectors(pc0)
    pc0_normals, pc0_tangent = get_orthogonal_vectors(pc0_vectors, surf.point_normals)
    pc0_tangent_mag = get_vector_magnitude(pc0_tangent)
logger.debug("Tangent vector pc0 {}".format(pc0_tangent_mag[0:2]))

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

logger.debug("Tangent vector pc1 {}".format(pc1_tangent_mag[0:2]))
logger.debug("Tangent vector pc2 {}".format(pc2_tangent_mag[0:2]))

# 8. Compute the gradients for every points
logger.info("Spline fitting and calculating gradients ...")
gradients = calculate_gradient(pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag, inward_distance, use_parabolic=parabolic_fitting)
logger.debug("gradients {}".format(gradients[0:2]))

# 9. Assign the gradients back to the surface
logger.info("Calculating WSS")
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
