''' WSS calculator - interpolation on velocity components
WSS calculator from Feridan 2020 https://github.com/EdwardFerdian/wss_mri_calculator/tree/master
Base implementation based on Petersson 2012 
"Assessment of the accuracy of MRI wall shear stress estimation using numerical simulations"
https://doi.org/10.1002/jmri.23610
'''

import pyvista as pv
import numpy as np
import logging
from wss_utils import *
import matplotlib.pyplot as plt 
from surfaceMesh import *
# from plot_functions import *

target_path = '/Users/jonathan/Documents/applications/superres/ubend_1000_m1/vtk_eval'
plot_path = target_path + '/plots'

cfd_file = target_path + '/cfd/cfd_wss_1.vtk'

# r_values = np.array([100,200,400,800])
# r_values = np.array([200,400,800,1600])
# r_values = np.array([200])
r = 100

# interpolated_grid_file = f'/data_HR_20_r{r}.h5'
interpolated_grid_file = f'/data_HR_r{r}.h5'

# extenstion name string to output png files
ext_string = '_HR'  # '_LR'

### inward distance point factors
# factors = np.array([1,2,3,4,5,6,7,8])
# factors = np.array([1,2,4,8])
factors = np.array([1])

# dynamic viscosity
viscosity = 3.18e-3
density = 1060

# show plot
show_plot = False # False, True
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


wss_means = []
wss_maxs = []

uy_spline_values = []
vy_spline_values = []
wy_spline_values = []

gradients_list = []
# factors to inward distance
for factor in factors:
    # input_filepath = target_path + f'/data_HR_r{r}.h5'
    input_filepath = target_path + '/' + interpolated_grid_file
    mask_filepath = input_filepath

    # input parameters
    voxel_size = r/1000 # in mm
    # Inward normal distance to sample points from wall (in mm)
    inward_distance = r/1000   # in mm
    
    # ----------- Parameters ----------------------
    # If the dimension is not isotropic, you might as well edit it here
    spacing = (voxel_size,voxel_size,voxel_size) # in mm
    inward_distance = factor*inward_distance
    smoothing_iteration = smoothing

    # 1. Load velocity vectors and segmentation/mask
    # Note: Please replace the loader depending on your data format, this is just an example
    # u,v,w = load_vector_fields(input_filepath, ['u', 'v', 'w'], 0)
    u,v,w = load_my_vector_fields(input_filepath, ['u', 'v', 'w'], 0)
    # Convert velocities in mm/s
    u,v,w = u*1000, v*1000, w*1000

    # print("shape u   ", u.shape)
    # print("shape v   ", v.shape)
    # print("shape w   ", w.shape)
    u,v,w = u.astype(np.float64), v.astype(np.float64), w.astype(np.float64)

    # mask = load_segmentation(mask_filepath, 'mask', 0)
    mask = load_my_segmentation(mask_filepath, 'mask', 0)
    mask = mask.astype(np.float64)

    # 2. Construct uniform grid for vector values and mask
    velocity = create_uniform_vector(u,v,w, spacing)
    mesh = create_uniform_grid(mask, spacing)
    mesh = mesh.threshold_percent(40)
    # mesh = mesh.threshold([0.3,0.8])

    # 3. Get the volume within the mask (for visualization purpose)
    volume = mesh.sample(velocity)
    volume.set_active_scalars("Velocity")

    # 4. Get the surface of the mesh
    # Note: do not use extract_geometry, the PyVista manual on the website is misleading
    surf = mesh.extract_surface()
    surf = surf.smooth(n_iter=smoothing_iteration)

    ### translate both meshes to same origin and scaling
    translation_surf = tuple(new_origin[i] - surf.center[i] for i in range(3))
    surf_trans = surf.translate(translation_surf)
    # find closest point to given target for analysis
    surf_target, surf_target_idx = find_closest_point(cfd_point_stenosis, surf_trans.points)
    
    # 5. Calculate the point normals (depends on what we computed on cells or on nodes)
    # Make sure we compute only the point normals, not the cell normals, they have the same array name
    surf.compute_normals(point_normals=True, cell_normals=True, inplace=True, flip_normals=True)

    # 6. Compute the coordinates from surface (pc0) to the equidistant points in the inward normal (pc1 and pc2)
    # Construct a polydata of points from the surface points
    pc0 = pv.PolyData(surf.points)
    pc0 = pc0.sample(velocity)
    pc0.set_active_scalars("Velocity")

    print("Voxel size        ", voxel_size)
    print("Inward distance   ", inward_distance)
    
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
    # gradients, x_, y_ = calculate_gradient(pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag, inward_distance, use_parabolic=parabolic_fitting)
    
    # Compute the gradients for every velocity component, for pc0 all zeros
    u_gradients, x_, uy_ = calculate_gradient(pc0_tangent_mag, pc1_tangent[:,0], pc2_tangent[:,0], inward_distance, use_parabolic=parabolic_fitting)
    v_gradients, x_, vy_ = calculate_gradient(pc0_tangent_mag, pc1_tangent[:,1], pc2_tangent[:,1], inward_distance, use_parabolic=parabolic_fitting)
    w_gradients, x_, wy_ = calculate_gradient(pc0_tangent_mag, pc1_tangent[:,2], pc2_tangent[:,2], inward_distance, use_parabolic=parabolic_fitting)
    
    # compute magntidue of gradient vector
    gradients_vec = np.vstack((u_gradients, v_gradients, w_gradients)).T
    gradients = np.linalg.norm(gradients_vec,axis=1)
    
    uy_spline_values.append(uy_)
    vy_spline_values.append(vy_)
    wy_spline_values.append(wy_)
    gradients_list.append(gradients)

    # target idx
    # plt.plot(x_, y_[surf_target_idx], '-o')
    # plt.xlabel('Inward Distance [mm]')
    # plt.ylabel('Velocity magnitude [mm/s]')
    # plt.show()
    
    for i in range(int(len(uy_)/1000)):
        plt.plot(x_, uy_[i*1000], '-o', label=f'u: dist={inward_distance}')
        plt.plot(x_, vy_[i*1000], '-o', label=f'v: dist={inward_distance}')
        plt.plot(x_, wy_[i*1000], '-o', label=f'w: dist={inward_distance}')
    plt.xlabel('Inward Distance [mm]')
    plt.ylabel('Velocity magnitude [mm/s]')
    # plt.savefig(plot_path + f'/velocity_fit_r{r}_f{factor}{ext_string}.png', dpi=200, bbox_inches='tight')
    plt.show()

    # 9. Assign the gradients back to the surface
    # gradients in 1/s
    wss = gradients * viscosity

    surf["wss"] = wss
    # use the pc1 vectors for visualization
    surf['wss_vectors'] = pc1_tangent

    print("Mean WSS [Pa]     ", round(np.mean(wss),8))
    print("Max WSS  [Pa]     ", round(np.max(wss),8), "\n")

    # append to list
    wss_means.append(np.mean(wss))
    wss_maxs.append(np.max(wss))
    
    # # plot histogram
    # histogram, bin_edges = np.histogram(wss.ravel(), bins='auto')
    # plt.figure(figsize=(8, 6))
    # plt.plot(bin_edges[0:-1], histogram)  # Bin edges have one extra element
    # # plt.title('Histogram')
    # plt.xlabel(f'WSS [Pa]')
    # plt.ylabel('Count')
    # plt.grid(True)
    # plt.show()

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
        p.add_text("WSS ({})".format(fitting_functext), font_size=20, color='black')
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

# restructure gradients list
gradients_array = np.array(gradients_list)

# ------------------------------------------------------------------
# domain analysis
# ------------------------------------------------------------------
# add zero point for cfd
# r_values_plot = np.insert(factors, 0, 0)/1000
factors_plot = np.insert(factors, 0, 0)
inward_distances = factors_plot*voxel_size

wss_means_plot = np.insert(wss_means, 0, cfd_wss_mean)
wss_maxs_plot = np.insert(wss_maxs, 0, cfd_wss_max)

plt.plot(inward_distances, wss_means_plot, '-o')
plt.xlabel('Inward Distance of 1st point [mm]')
plt.ylabel('Mean WSS [Pa]')
plt.grid(True)
plt.savefig(plot_path + f'/wss_mean_inward_distance_r{r}{ext_string}.png', dpi=200, bbox_inches='tight')
plt.show()

plt.plot(inward_distances, wss_maxs_plot, '-o')
plt.xlabel('Inward Distance of 1st point [mm]')
plt.ylabel('Max WSS [Pa]')
plt.grid(True)
plt.savefig(plot_path + f'/wss_max_inward_distance_r{r}{ext_string}.png', dpi=200, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# target point analysis
# ------------------------------------------------------------------
# velocity spline fit
colors = ['red', 'blue', 'green', 'magenta']
for i, factor in enumerate(factors):
    plt.plot(x_*factor, uy_spline_values[i][surf_target_idx], '-', color=colors[i], label=f'u: dist {round(factor*voxel_size,1)}')
    plt.plot(x_*factor, vy_spline_values[i][surf_target_idx], '-', color=colors[i], label=f'v: dist {round(factor*voxel_size,1)}')
    plt.plot(x_*factor, wy_spline_values[i][surf_target_idx], '-', color=colors[i], label=f'w: dist {round(factor*voxel_size,1)}')
plt.xlabel('Inward Distance [mm]')
plt.ylabel('Velocity magnitude [mm/s]')
plt.title(f'Velocity magnitude fit for {voxel_size} mm')
plt.legend()
plt.savefig(target_point_path + f'/velocity_fit_factors_r{r}{ext_string}.png', dpi=200, bbox_inches='tight')
plt.show()


# evaluation of WSS gradient at target
cfd_target_wss_vec = cfd_mesh.point_data['wallShearStress'][cfd_target_idx]

cfd_target_wss = np.linalg.norm(cfd_target_wss_vec)
gradients_plot = np.insert(gradients_array[:,surf_target_idx]*viscosity, 0, cfd_target_wss*density)

plt.plot(inward_distances,gradients_plot, 'o-')

plt.xlabel('Inward Distance [mm]')
plt.ylabel('WSS [Pa]')
plt.title(f'WSS in target point r={r/1000}mm')
plt.legend()
plt.grid()
plt.savefig(target_point_path + f'/wss_target_point_{r}{ext_string}.png', dpi=200, bbox_inches='tight')
plt.show()

1/0
# plot cfd mesh and surf_trans
p = pv.Plotter(notebook=0)
p.add_mesh(cfd_mesh, color='blue', opacity=0.1)
p.add_mesh(surf_trans, color='red', show_edges=True, opacity=0.1)
p.add_mesh(pv.PolyData(surf_target), color='red')
p.add_mesh(pv.PolyData(cfd_target_pts), color='blue')
# p.add_mesh(pv.PolyData(surf_target), color='red')
p.show_axes() 
p.show_grid()
p.show()