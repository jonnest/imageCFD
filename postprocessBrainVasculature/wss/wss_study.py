''' WSS calculator 
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
# from plot_functions import *

target_path = '/Users/jonathan/Documents/applications/superres/ubend_1000_v0/vtk_eval'

# dynamic viscosity
viscosity = 3.18e-3

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

# set the logging level on the imported script
logging.getLogger("wss_utils").setLevel(loglevel)  

logging.basicConfig()
logger = logging.getLogger('calculate_wss')
logger.setLevel(loglevel)

# ------------------------------------------------------------------
cfd_wss_max = 210.82
cfd_wss_mean = 18.36


# r_values = np.array([100,200,400,800])
# r_values = np.array([200,400,800,1600])
r_values = np.array([200])

# extenstion name string to output png files
ext_string = '_LR'  # '_LR'

wss_means = []
wss_maxs = []

for r in r_values:
    # input_filepath = target_path + f'/data_HR_r{r}.h5'
    input_filepath = target_path + f'/data_LR_20_r{r}.h5'
    mask_filepath = input_filepath

    # input parameters
    voxel_size = r/1000 # in mm
    # Inward normal distance to sample points from wall (in mm)
    inward_distance = r/1000   # in mm
    print("\nVoxel size        ", voxel_size)
    print("Inward distance   ", voxel_size)
    
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

    # append to list
    wss_means.append(np.mean(wss))
    wss_maxs.append(np.max(wss))
    
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


### plot evaluation

# add zero point for cfd
r_values_plot = np.insert(r_values, 0, 0)/1000

wss_means_plot = np.insert(wss_means, 0, cfd_wss_mean)
wss_maxs_plot = np.insert(wss_maxs, 0, cfd_wss_max)

plt.plot(r_values_plot, wss_means_plot, '-o')
plt.xlabel('Voxel size [mm]')
plt.ylabel('Mean WSS [Pa]')
plt.grid(True)
plt.savefig(target_path + f'/wss_mean{ext_string}.png', dpi=200, bbox_inches='tight')
plt.show()

plt.plot(r_values_plot, wss_maxs_plot, '-o')
plt.xlabel('Voxel size [mm]')
plt.ylabel('Max WSS [Pa]')
plt.grid(True)
plt.savefig(target_path + f'/wss_max{ext_string}.png', dpi=200, bbox_inches='tight')
plt.show()