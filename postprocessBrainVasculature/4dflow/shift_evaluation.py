import os,sys
import h5py
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import pyacvd
from surfaceMesh import *
import skimage
from skimage.filters import threshold_multiotsu
from scipy.ndimage import median_filter
from scipy import ndimage
import time
import vtk
import pymeshfix
import nibabel as nib
from scipy.ndimage import binary_erosion

from fileOperators import *
from flow_field_metrics import compute_vorticity, disparity_density, spatial_cross_correlation


w_dir = '/Users/jonathan/Documents/eth/02_conferences/ismrm_2025/vein_ismrm/'

plot_folder = 'plots'

image_res = 0.8e-3
venc = 0.3

scalars = 'data' # 'scalars' 

loader = loadData()

# read segmentation mask
mask_pv = pv.read(w_dir + '/vol000/mask.vtk')
mask = np.reshape(mask_pv.point_data[scalars], mask_pv.dimensions, order='F')

# ext string in case of mask erosion
erosion_string = '_eroded' # ''
mask = binary_erosion(mask, iterations=1)

steps = np.linspace(-25,25,21)
nr_shifts = len(steps)

# ------------------------------------------------------------------
# right sigmoid
sig_r_coord = [125,140,12] # [125,140,12] , [125,140,12]
sig_r_dims = [20,25,25]    # [25,25,25],  [30,36,30]
# left sigmoid
sig_l_coord = [125,70,10]
sig_l_dims = [20,25,25]
# saggital superior sinus
sss_coord = [10,105,0]
sss_dims = [40,25,50]
# transverse right sinus
trans_r_coord = [65,165,10]
trans_r_dims = [70,25,25]
# transverse left sinus
trans_l_coord = [65,45,10]
trans_l_dims = [70,25,25]
# jugular right sinus
jug_r_coord = [150,145,12]
jug_r_dims = [30,25,45]
# jugular left sinus
jug_l_coord = [150,65,12]
jug_l_dims = [30,25,45]

# generate mask boxes

def mask_box(mask, box_coord, box_dims):
    mask_box = np.zeros(mask.shape, dtype=bool)
    mask_box[box_coord[0]:box_coord[0]+box_dims[0],
             box_coord[1]:box_coord[1]+box_dims[1],
             box_coord[2]:box_coord[2]+box_dims[2]] = mask[box_coord[0]:box_coord[0]+box_dims[0],
                                                           box_coord[1]:box_coord[1]+box_dims[1],
                                                           box_coord[2]:box_coord[2]+box_dims[2]]
    return mask_box

mask_sss = mask_box(mask, sss_coord, sss_dims)
mask_sig_l = mask_box(mask, sig_l_coord, sig_l_dims)
mask_sig_r = mask_box(mask, sig_r_coord, sig_r_dims)
mask_trans_l = mask_box(mask, trans_l_coord, trans_l_dims)
mask_trans_r = mask_box(mask, trans_r_coord, trans_r_dims)
mask_jug_l = mask_box(mask, jug_l_coord, jug_l_dims)
mask_jug_r = mask_box(mask, jug_r_coord, jug_r_dims)

masks_roi = [mask_sss, mask_trans_l, mask_trans_r,  mask_sig_l, mask_sig_r, mask_jug_l, mask_jug_r]
masks_roi_str = ['SSS', 'left transverse', 'right transverse', 'left sigmoid', 'right sigmoid', 'left jugular', 'right jugular']

# ------------------------------------------------------------------
vor1_meanshift = []
vor2_meanshift = []
vor1_maxshift = []
vor2_maxshift = []

vor1_all_means = np.zeros((nr_shifts, 14))
vor2_all_means = np.zeros((nr_shifts, 14))
vor1_all_maxs = np.zeros((nr_shifts, 14))
vor2_all_maxs = np.zeros((nr_shifts, 14))

vor1_mean_masks = np.zeros((len(masks_roi), len(steps)))
vor2_mean_masks = np.zeros((len(masks_roi), len(steps)))

ske1_all_means = np.zeros((nr_shifts, 14))
ske2_all_means = np.zeros((nr_shifts, 14))

ske1_mean_masks = np.zeros((len(masks_roi), len(steps)))
ske2_mean_masks = np.zeros((len(masks_roi), len(steps)))

flow_rate1_means = []
flow_rate2_means = []

corr = []

for i in range(len(steps)):
    image1 = loader.loadFullFlowField(w_dir + f'/recons/shift{steps[i]:.1f}_bin0.h5')
    image2 = loader.loadFullFlowField(w_dir + f'/recons/shift{steps[i]:.1f}_bin1.h5')

    if image1.shape[3]==4:
        image1 = image1[:, :, :, 1:, :]
    
    if image2.shape[3]==4:
        image2 = image2[:, :, :, 1:, :]
    
    # drop reference scan without velocity encoding
    image1 = np.squeeze(np.array(image1))
    image2 = np.squeeze(np.array(image2))
    
    u_field1 = np.angle(image1)*venc/np.pi
    u_field2 = np.angle(image2)*venc/np.pi

    # disparity    
    # bin_centers1, counts1 = disparity_density(u_field1, mask)
    # bin_centers2, counts2 = disparity_density(u_field2, mask)
    # plt.plot(bin_centers1, counts1, lw=1)
    # plt.plot(bin_centers2, counts2, lw=1)
    # plt.show()
    
    # compiute magnitude
    u_field1_norm = np.sqrt(np.sum(u_field1**2,axis=-2))
    u_field2_norm = np.sqrt(np.sum(u_field2**2,axis=-2))
    
    # for each shift - collect means over time frames
    vor1_means = []
    vor2_means = []
    
    # temporary means for 1 shift over time
    vor1_mean_masks_tmp = np.zeros((len(masks_roi), u_field1.shape[-1]))
    vor2_mean_masks_tmp = np.zeros((len(masks_roi), u_field1.shape[-1]))

    ske1_mean_masks_tmp = np.zeros((len(masks_roi), u_field1.shape[-1]))
    ske2_mean_masks_tmp = np.zeros((len(masks_roi), u_field1.shape[-1]))
    
    # flow rates 
    flow_factor = 1e6
    z_slice = 10
    flow_rate1 = []
    flow_rate2 = []
    
    for t in range(u_field1.shape[-1]):
        # mask field - keep shape - for each time frame
        u_field1_shape_masked = np.where(mask, u_field1_norm[..., t], 0)
        u_field2_shape_masked = np.where(mask, u_field2_norm[..., t], 0)
    
        flow_rate1.append(np.sum(u_field1_shape_masked[...,z_slice])*image_res**2*flow_factor)
        flow_rate2.append(np.sum(u_field2_shape_masked[...,z_slice])*image_res**2*flow_factor)
        
        vor1_mag, _ = compute_vorticity(u_field1[...,t], image_res)
        vor2_mag, _ = compute_vorticity(u_field2[...,t], image_res)

        vor1_means.append(np.mean(vor1_mag[mask.astype(bool)]))
        vor2_means.append(np.mean(vor2_mag[mask.astype(bool)]))
        
        vor1_all_means[i,t] = np.mean(vor1_mag[mask.astype(bool)])
        vor2_all_means[i,t] = np.mean(vor2_mag[mask.astype(bool)])
        vor1_all_maxs[i,t] = np.max(vor1_mag[mask.astype(bool)])
        vor2_all_maxs[i,t] = np.max(vor2_mag[mask.astype(bool)])
        
        ske1_all_means[i,t] = np.mean(0.5 * np.linalg.norm(u_field1[mask.astype(bool),:,t], axis=1)**2)
        ske2_all_means[i,t] = np.mean(0.5 * np.linalg.norm(u_field2[mask.astype(bool),:,t], axis=1)**2)
        
        for m in range(len(masks_roi)):
            ske1_mean_masks_tmp[m,t] = np.mean(0.5 * np.linalg.norm(u_field1[masks_roi[m].astype(bool),:,t], axis=1)**2)
            ske2_mean_masks_tmp[m,t] = np.mean(0.5 * np.linalg.norm(u_field2[masks_roi[m].astype(bool),:,t], axis=1)**2)
 
            if m!=0 and m!=5 and m!=6 :
                vor1_mean_masks_tmp[m,t] = np.mean(vor1_mag[masks_roi[m].astype(bool)])
                vor2_mean_masks_tmp[m,t] = np.mean(vor2_mag[masks_roi[m].astype(bool)])
            # for superior saggtial sinus exclude outliers
            elif m==0 or m==5 or m==6:
                vor1_mag_ = vor1_mag[masks_roi[m].astype(bool)]
                vor2_mag_ = vor2_mag[masks_roi[m].astype(bool)]
                
                q1_1 = np.percentile(vor1_mag_, 25)
                q3_1 = np.percentile(vor1_mag_, 75)
                lower_bound_1 = q1_1 - 1.5 * (q3_1 - q1_1)
                upper_bound_1 = q3_1 + 1.5 * (q3_1 - q1_1)

                q1_2 = np.percentile(vor2_mag_, 25)
                q3_2 = np.percentile(vor2_mag_, 75)
                lower_bound_2 = q1_2 - 1.5 * (q3_2 - q1_2)
                upper_bound_2 = q3_2 + 1.5 * (q3_2 - q1_2)
                
                vor1_mag_ = vor1_mag_[(vor1_mag_ >= lower_bound_1) & (vor1_mag_ <= upper_bound_1)]
                vor2_mag_ = vor2_mag_[(vor2_mag_ >= lower_bound_2) & (vor2_mag_ <= upper_bound_2)]

                vor1_mean_masks_tmp[m,t] = np.mean(vor1_mag_)
                vor2_mean_masks_tmp[m,t] = np.mean(vor2_mag_)
                
                
            
    flow_rate1_means.append(np.mean(flow_rate1))
    flow_rate2_means.append(np.mean(flow_rate2))

    vor1_meanshift.append(np.mean(vor1_means))
    vor2_meanshift.append(np.mean(vor2_means))
    vor1_maxshift.append(np.max(vor1_means))
    vor2_maxshift.append(np.max(vor2_means))

    # take mean over time for each shift and each mask_roi
    for m in range(len(masks_roi)):
        vor1_mean_masks[m,i] = np.mean(vor1_mean_masks_tmp[m,:])
        vor2_mean_masks[m,i] = np.mean(vor2_mean_masks_tmp[m,:])

        ske1_mean_masks[m,i] = np.mean(ske1_mean_masks_tmp[m,:])
        ske2_mean_masks[m,i] = np.mean(ske2_mean_masks_tmp[m,:])
        
    # masked flow field for all time frames
    u_field1_masked = u_field1[mask.astype(bool),...]
    u_field2_masked = u_field2[mask.astype(bool),...]


# write list to np array
flow_rate1_means = np.array(flow_rate1_means)
flow_rate2_means = np.array(flow_rate2_means)

vor1_meanshift = np.array(vor1_meanshift)
vor2_meanshift = np.array(vor2_meanshift)

vor1_maxshift = np.array(vor1_maxshift)
vor2_maxshift = np.array(vor2_maxshift)


# ------------------------------------------------------------------
# plots
# ------------------------------------------------------------------
# Plotting the vorticity field
# vor1_mag, vor1_vec = compute_vorticity(u_field1[...,t], image_res)
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.quiver(u_field1.shape[0], u_field1.shape[1], u_field1.shape[2], vor1_vec[...,0], vor1_vec[...,1], vor1_vec[...,2], length=0.3, normalize=True)
# # Set plot labels
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Vorticity Field Visualization')
# plt.show()


# Flatten grid points and vectors for PyVista
vor1_mag, vor1_vec = compute_vorticity(u_field1[...,t], image_res)

# crop image 
vor1_vec = vor1_vec[10:180, 45:185, 5:-5]
u_field1_crop = u_field1[10:180, 45:185, 5:-5]

nx, ny, nz = u_field1_crop.shape[:3]
# vor1_vec = np.where(mask, vor1_vec[...,:], 0)
mask_eroded = binary_erosion(mask, iterations=1)
vor1_x = np.where(mask_eroded[10:180, 45:185, 5:-5], vor1_vec[...,0], 0)
vor1_y = np.where(mask_eroded[10:180, 45:185, 5:-5], vor1_vec[...,1], 0)
vor1_z = np.where(mask_eroded[10:180, 45:185, 5:-5], vor1_vec[...,2], 0)

# threshold
threshold_value = 200
vor1_x = np.where(vor1_x > threshold_value, threshold_value, vor1_x)
vor1_y = np.where(vor1_y > threshold_value, threshold_value, vor1_y)
vor1_z = np.where(vor1_z > threshold_value, threshold_value, vor1_z)

vor1_x = np.where(vor1_x < -threshold_value, -threshold_value, vor1_x)
vor1_y = np.where(vor1_y < -threshold_value, -threshold_value, vor1_y)
vor1_z = np.where(vor1_z < -threshold_value, -threshold_value, vor1_z)

x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')

points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
# vectors = np.column_stack((vor1_vec[...,0].ravel(), vor1_vec[...,1].ravel(), vor1_vec[...,2].ravel()))
vectors = np.column_stack((vor1_x.ravel(), vor1_y.ravel(), vor1_z.ravel()))

# Normalize vectors to unit length
magnitudes = np.linalg.norm(vectors, axis=1)
# Avoid division by zero by setting zero-magnitude vectors to have unit length
magnitudes[magnitudes == 0] = 1
normalized_vectors = vectors / magnitudes[:, np.newaxis].max()

mesh = pv.PolyData(points)
mesh['vectors'] = normalized_vectors
# # Make a geometric object to use as the glyph
# geom = pv.Arrow()  # This could be any dataset

# Create glyphs with controlled scaling
# pv.set_jupyter_backend('trame')
glyphs = mesh.glyph(orient='vectors', scale='vectors', factor=15)# , geom=geom)  # Adjust factor to keep arrows within bounds
# Plot with constrained boundaries
# plotter = pv.Plotter()
plotter = pv.Plotter(notebook=True)
# plotter.add_mesh(glyphs, color='blue')
plotter.add_mesh(glyphs, show_scalar_bar=False, lighting=False, cmap='plasma')
plotter.add_scalar_bar(title='Normalized Vorticity [1/s]', n_labels=5)
# plotter.show_grid()
plotter.camera_position = 'yx'
# plotter.camera_position = [(0, -400, 150), (0, 0, 0), (0, 0, -1)]
# (400,300,150)
# plotter.camera_position = [(-400, 350, 150), ( 0,0,0 ), (0, 0, 0)]  # (position, focal point, view up vector)
# plotter.camera_position = [(330, 330, -180), ( 130, 130, 30 ), (-0.43, 0.43, -0.78)]  # (position, focal point, view up vector)

plotter.show(title="3D Vorticity Field Visualization")
plotter.screenshot(w_dir + f'/{plot_folder}/vorticity_field_{erosion_string}_thres{threshold_value}.png')



# grid = pv.ImageData()
# grid.dimensions = u_field1[10:180, 45:185, 5:55,...].shape[:3]
# # grid['velocity'] = np.column_stack((
# #     u_field1[10:180, 45:185, 5:55, 0, t].ravel(),
# #     u_field1[10:180, 45:185, 5:55, 1, t].ravel(),
# #     u_field1[10:180, 45:185, 5:55, 2, t].ravel()
# # ))
# grid['u'] = np.reshape(u_field1[10:180, 45:185, 5:55, :, t].astype('double'),-1,order='F')
# grid.save(os.path.join(w_dir, f'field_.vtk'))


# ------------------------------------------------------------------
# add the periodicity
# x=-25 to 0 corresponds to x=25 to 50 with y=-value
x_l = np.arange(-50, -27.4, 2.5)
x_r = np.arange(27.5, 50.1, 2.5)

xf_l = np.arange(-50, -24.9, 2.5)
xf_r = np.arange(25, 50.1, 2.5)

steps_period = np.arange(-50, 50.1, 2.5)
# ------------------------------------------------------------------
# flow_rate = (flow_rate1_means-flow_rate2_means)/flow_rate2_means*100
# plt.plot(steps, flow_rate) #, label='right sigmoid')
# plt.plot(x_l, -flow_rate[len(x_l)-1:], 'b', alpha=0.3)
# plt.plot(x_r, -flow_rate[:len(x_r)], 'b', alpha=0.3)
# plt.xlabel('Shifts')
# plt.ylabel(r'Flow rate $Diff_{Ex-In}$ [%]')
# plt.title('Flow rate')
# # plt.legend()
# plt.grid()
# plt.show()

# ------------------------------------------------------------------
vor_means = (vor1_meanshift-vor2_meanshift)/vor2_meanshift*100
plt.plot(steps, vor_means) #, label='right sigmoid')
plt.plot(xf_l, -vor_means[len(xf_l)-1:], 'b', alpha=0.3)
plt.plot(xf_r, -vor_means[:len(xf_r)], 'b', alpha=0.3)
plt.xlabel('Shifts')
plt.ylabel(r'Vorticity $Diff_{Ex-In}$ [%]')
plt.title('Shifted bins - mean of mean vorticity')
plt.legend()
plt.grid()
plt.savefig(w_dir + '/' + f'{plot_folder}/mean_vorticity_shifts{erosion_string}.png', dpi=400, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# colors = ['blue', 'cyan', 'grey', 'purple', 'magenta', 'green', 'lightgreen']
# for m in range(len(masks_roi)):
#     if m!=2:
#         vor_means = (vor1_mean_masks[m,:]- vor2_mean_masks[m,:])/vor2_mean_masks[m,:]*100
        
#         vor_means_period = np.concatenate((-vor_means[len(x_l)+1:], vor_means, -vor_means[:len(x_r)]))
        
#         plt.plot(steps, vor_means, color=colors[m], label=masks_roi_str[m])
#         plt.plot(xf_l, -vor_means[len(xf_l)-1:], color=colors[m], alpha=0.5)
#         plt.plot(xf_r, -vor_means[:len(xf_r)], color=colors[m], alpha=0.5)
#         plt.xlabel('Shifts')
#         plt.ylabel(r'Vorticity $Diff_{Ex-In}$ [%]')
#         plt.title('Shifted bins - mean of mean vorticity')
#         plt.legend()
#         plt.grid()
    
# plt.savefig(w_dir + '/' + f'{plot_folder}/mean_vorticity_shifts_roi{erosion_string}.png', dpi=400, bbox_inches='tight')
# plt.show()

# save data
np.save(w_dir + f'/data/vorticity_mean_bin0.npy', vor1_mean_masks)
np.save(w_dir + f'/data/vorticity_mean_bin1.npy', vor2_mean_masks)
for m in range(len(masks_roi)):
    np.save(w_dir + f'/data/vorticity_mean_bin0_mask{m}.npy', vor1_mean_masks[m,:])
    np.save(w_dir + f'/data/vorticity_mean_bin1_mask{m}.npy', vor2_mean_masks[m,:])


colors = ['blue', 'cyan', 'grey', 'purple', 'magenta', 'green', 'lightgreen']
for m in range(len(masks_roi)):
    if m!=2:
        vor_means = (vor1_mean_masks[m,:]- vor2_mean_masks[m,:])/vor2_mean_masks[m,:]*100        
        vor_means_period = np.concatenate((-vor_means[len(x_l)+1:], vor_means, -vor_means[:len(x_r)]))
        
        plt.plot(steps, vor_means, color=colors[m], label=masks_roi_str[m])
        plt.plot(xf_l, -vor_means[len(xf_l)-1:], color=colors[m], alpha=0.5)
        plt.plot(xf_r, -vor_means[:len(xf_r)], color=colors[m], alpha=0.5)
        plt.xlabel('Shifts')
        plt.ylabel(r'Vorticity $Diff_{Ex-In}$ [%]')
        plt.title('Shifted bins - mean of mean vorticity')
        plt.legend()
        plt.grid()
    
plt.savefig(w_dir + '/' + f'{plot_folder}/mean_vorticity_shifts_roi{erosion_string}.png', dpi=400, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# Create a figure and a row of 4 subplots
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
# Plot in each subplot
location = ['left', 'right']
linestyle_str = ['solid', 'dashed']
for i, m in enumerate([0]):
    vor_means = (vor1_mean_masks[m,:]- vor2_mean_masks[m,:])/vor2_mean_masks[m,:]*100
    axs[0].plot(steps, vor_means, color='green', label=masks_roi_str[m], linestyle=linestyle_str[i])
    axs[0].plot(xf_l, -vor_means[len(xf_l)-1:], color='green', alpha=0.5, linestyle=linestyle_str[i])
    axs[0].plot(xf_r, -vor_means[:len(xf_r)], color='green', alpha=0.5, linestyle=linestyle_str[i])
    axs[0].set_title('SSS')

for i, m in enumerate([1,2]):
    vor_means = (vor1_mean_masks[m,:]- vor2_mean_masks[m,:])/vor2_mean_masks[m,:]*100
    axs[1].plot(steps, vor_means, color='green', label=location[i], linestyle=linestyle_str[i])
    axs[1].plot(xf_l, -vor_means[len(xf_l)-1:], color='green', alpha=0.5, linestyle=linestyle_str[i])
    axs[1].plot(xf_r, -vor_means[:len(xf_r)], color='green', alpha=0.5, linestyle=linestyle_str[i])
    axs[1].set_title('Transverse Sinus')

for i, m in enumerate([3,4]):
    vor_means = (vor1_mean_masks[m,:]- vor2_mean_masks[m,:])/vor2_mean_masks[m,:]*100
    axs[2].plot(steps, vor_means, color='green', label=location[i], linestyle=linestyle_str[i])
    axs[2].plot(xf_l, -vor_means[len(xf_l)-1:], color='green', alpha=0.5, linestyle=linestyle_str[i])
    axs[2].plot(xf_r, -vor_means[:len(xf_r)], color='green', alpha=0.5, linestyle=linestyle_str[i])
    axs[2].set_title('Sigmoid Sinus')

for i, m in enumerate([5,6]):
    vor_means = (vor1_mean_masks[m,:]- vor2_mean_masks[m,:])/vor2_mean_masks[m,:]*100
    axs[3].plot(steps, vor_means, color='green', label=location[i], linestyle=linestyle_str[i])
    axs[3].plot(xf_l, -vor_means[len(xf_l)-1:], color='green', alpha=0.5, linestyle=linestyle_str[i])
    axs[3].plot(xf_r, -vor_means[:len(xf_r)], color='green', alpha=0.5, linestyle=linestyle_str[i])
    axs[3].set_title('Internal Jugular')

plt.tight_layout()  # Adjust spacing between subplots
plt.savefig(w_dir + '/' + f'{plot_folder}/mean_vorticity_shifts_roi_rows{erosion_string}_mod.png', dpi=400, bbox_inches='tight')
plt.show()


# ------------------------------------------------------------------
# vor_maxs = (vor1_maxshift-vor2_maxshift)/vor2_maxshift*100
# plt.plot(steps, vor_maxs) #, label='right sigmoid')
# plt.plot(x_l, -vor_maxs[len(x_l)-1:], 'b', alpha=0.3)
# plt.plot(x_r, -vor_maxs[:len(x_r)], 'b', alpha=0.3)
# plt.xlabel('Shifts')
# plt.ylabel(r'Vorticity $Diff_{Ex-In}$ [%]')
# plt.title('Shifted bins - max of mean vorticity')
# plt.legend()
# plt.show()

# ------------------------------------------------------------------
# plt.plot(steps, corr)
# plt.plot(x_l, -corr[len(x_l)-1:], 'b', alpha=0.3)
# plt.plot(x_r, -corr[:len(x_r)], 'b', alpha=0.3)
# plt.title("Cross correlation between fields")
# plt.show()

# ------------------------------------------------------------------
# # flatten 'C' : row-major, after shifts, 'F' : column major - over time frames
# plt.plot(vor1_all_means.flatten('F')/vor2_all_means.flatten('F')) #, label='right sigmoid')
# plt.xlabel('Shifts')
# plt.ylabel('Vorticity [1/s]')
# plt.grid()
# plt.title('Mean vorticity over all shifts and time frames')
# plt.legend()
# plt.show()

# ------------------------------------------------------------------
ske_means = (np.mean(ske1_all_means,axis=1) - np.mean(+ske2_all_means,axis=1))/np.mean(ske2_all_means,axis=1)*100
plt.plot(steps, ske_means) #, label='right sigmoid')
plt.plot(xf_l, -ske_means[len(xf_l)-1:], 'b', alpha=0.3)
plt.plot(xf_r, -ske_means[:len(xf_r)], 'b', alpha=0.3)
plt.xlabel('Shifts')
# plt.ylabel(r'Specific kinetic energy $\frac{m^2}{s^2}$')
plt.ylabel(r'Mean specific kinetic energy $Diff_{Ex-In}$ [%]')
plt.title('Mean specific kinetic energy')
# plt.legend()
plt.grid()
plt.savefig(w_dir + '/' + f'{plot_folder}/mean_ske_shifts{erosion_string}.png', dpi=400, bbox_inches='tight')
plt.show()

# save data
np.save(w_dir + f'/data/ske_mean_bin0.npy', ske1_mean_masks)
np.save(w_dir + f'/data/ske_mean_bin1.npy', ske2_mean_masks)
for m in range(len(masks_roi)):
    np.save(w_dir + f'/data/ske_mean_bin0_mask{m}.npy', ske1_mean_masks[m,:])
    np.save(w_dir + f'/data/ske_mean_bin1_mask{m}.npy', ske2_mean_masks[m,:])


colors = ['blue', 'cyan', 'grey', 'purple', 'magenta', 'green', 'lightgreen']
for m in range(len(masks_roi)):
    # if m!=2:
    ske_means = (ske1_mean_masks[m,:]- ske2_mean_masks[m,:])/ske2_mean_masks[m,:]*100
    
    ske_means_period = np.concatenate((-ske_means[len(x_l)+1:], ske_means, -ske_means[:len(x_r)]))
    
    plt.plot(steps, ske_means, color=colors[m], label=masks_roi_str[m])
    plt.plot(xf_l, -ske_means[len(xf_l)-1:], color=colors[m], alpha=0.5)
    plt.plot(xf_r, -ske_means[:len(xf_r)], color=colors[m], alpha=0.5)
    plt.xlabel('Shifts')
    plt.ylabel(r'Kinetic energy $Diff_{Ex-In}$ [%]')
    plt.title('Shifted bins - mean of mean kinetic energy')
    plt.legend()
    plt.grid()
    
plt.savefig(w_dir + '/' + f'{plot_folder}/mean_ske_shifts_roi{erosion_string}.png', dpi=400, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# Create a figure and a row of 4 subplots
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
# Plot in each subplot
location = ['left', 'right']
linestyle_str = ['solid', 'dashed']
for i, m in enumerate([0]):
    ske_means = (ske1_mean_masks[m,:]- ske2_mean_masks[m,:])/ske2_mean_masks[m,:]*100
    axs[0].plot(steps, ske_means, color='green', label=masks_roi_str[m], linestyle=linestyle_str[i])
    axs[0].plot(xf_l, -ske_means[len(xf_l)-1:], color='green', alpha=0.5, linestyle=linestyle_str[i])
    axs[0].plot(xf_r, -ske_means[:len(xf_r)], color='green', alpha=0.5, linestyle=linestyle_str[i])
    axs[0].set_title('SSS')

for i, m in enumerate([1,2]):
    ske_means = (ske1_mean_masks[m,:]- ske2_mean_masks[m,:])/ske2_mean_masks[m,:]*100
    axs[1].plot(steps, ske_means, color='green', label=location[i], linestyle=linestyle_str[i])
    axs[1].plot(xf_l, -ske_means[len(xf_l)-1:], color='green', alpha=0.5, linestyle=linestyle_str[i])
    axs[1].plot(xf_r, -ske_means[:len(xf_r)], color='green', alpha=0.5, linestyle=linestyle_str[i])
    axs[1].set_title('Transverse Sinus')

for i, m in enumerate([3,4]):
    ske_means = (ske1_mean_masks[m,:]- ske2_mean_masks[m,:])/ske2_mean_masks[m,:]*100
    axs[2].plot(steps, ske_means, color='green', label=location[i], linestyle=linestyle_str[i])
    axs[2].plot(xf_l, -ske_means[len(xf_l)-1:], color='green', alpha=0.5, linestyle=linestyle_str[i])
    axs[2].plot(xf_r, -ske_means[:len(xf_r)], color='green', alpha=0.5, linestyle=linestyle_str[i])
    axs[2].set_title('Sigmoid Sinus')

for i, m in enumerate([5,6]):
    ske_means = (ske1_mean_masks[m,:]- ske2_mean_masks[m,:])/ske2_mean_masks[m,:]*100
    axs[3].plot(steps, ske_means, color='green', label=location[i], linestyle=linestyle_str[i])
    axs[3].plot(xf_l, -ske_means[len(xf_l)-1:], color='green', alpha=0.5, linestyle=linestyle_str[i])
    axs[3].plot(xf_r, -ske_means[:len(xf_r)], color='green', alpha=0.5, linestyle=linestyle_str[i])
    axs[3].set_title('Internal Jugular')

# Display the plot
plt.tight_layout()  # Adjust spacing between subplots
plt.savefig(w_dir + '/' + f'{plot_folder}/mean_ske_shifts_roi_rows{erosion_string}.png', dpi=400, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# ske_maxs = (np.max(ske1_all_means,axis=1) - np.max(ske2_all_means,axis=1))/np.max(ske2_all_means,axis=1)*100
# plt.plot(steps, ske_maxs) #, label='right sigmoid')
# plt.plot(x_l, -ske_maxs[len(x_l)-1:], 'b', alpha=0.3)
# plt.plot(x_r, -ske_maxs[:len(x_r)], 'b', alpha=0.3)
# plt.xlabel('Shifts')
# # plt.ylabel(r'Specific kinetic energy $\frac{m^2}{s^2}$')
# plt.ylabel(r'Specific kinetic energy $Diff_{Ex-In}$ [%]')
# plt.title('Max specific kinetic energy')
# # plt.legend()
# plt.grid()
# plt.show()