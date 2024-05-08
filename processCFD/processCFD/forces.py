import sys,os,re
import numpy as np
sys.path.append('/Users/jonathan/Documents/code/pytilities/utilities')
from mytools import *
import pandas as pd
import matplotlib.pyplot as plt
import csv



### data folder
w_dir = '/Users/jonathan/Documents/applications/cfd/cerebral_veins/'
# foam_folder = 'tinnitus_240110_seg1_les_m2.4_analysis'
foam_folder = 'mesh_study_tinnitus_240110/tinnitus_240110_seg1_les_m05_dt'

target = w_dir + foam_folder

# probe_file = 'bulge_box_time.csv'
# probe_file = 'box_new_time.csv'
probe_file = 'bnd3.csv'


file_path = os.path.join(target,probe_file)

# df = pd.read_csv(file_path)
data = np.loadtxt(file_path, skiprows=1, delimiter=',')

### Perform FFT - velocity and pressure

t_redund = np.unique(data[:,1])
time_step = t_redund[1]-t_redund[0]



############################################################################
### fft over all time steps and spatial coordinates
############################################################################
print('\n--- fft over all time steps and spatial coordinates ---\n')

signals = [ data[:,5], data[:,6], data[:,7] ]

plt.figure(figsize=(10, 5))
i = 0
for signal in signals:
    fft_output = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), 1/len(signal))  # Frequency bins
    # Plot
    plt.plot(fft_freq[:len(fft_freq)//2], np.abs(fft_output[:len(fft_output)//2]), label=f'force:{i}')  # Plot only positive frequencies
    i += 1
    
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
#plt.xlim([0,20])
plt.title('FFT of the Signal')
plt.savefig(target+'/fft_probe_up.png', dpi=200)
plt.show()


# ############################################################################
# print('\n--- all spatial points over time dimension ---\n')

# nr_points = int(len(data)/len(t_redund))

# # loop over all spatial points in the first timestep

# # for t in range(len(t_redund)):
# for t in range(40):
#     bool_arr = data[:,2:5]==data[t,2:5]
#     selector = np.all(bool_arr, axis=1)

#     point_over_time = data[selector]
    
#     force_mag =  np.linalg.norm((point_over_time[:,5],
#                                         point_over_time[:,6],
#                                         point_over_time[:,7]), axis=0)
#     plt.plot(force_mag )

# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()

############################################################################
### 1 spatial point over time dimension
############################################################################
print('\n--- FFT : 1 spatial point over time dimension ---\n')
# get one point over multiple time steps
t = 40
bool_arr = data[:,2:5]==data[t,2:5]
selector = np.all(bool_arr, axis=1)

point_over_time = data[np.all(bool_arr, axis=1)]
# fft_output = np.fft.fft(point_over_time[:,5]   )
fft_output = np.fft.fft(np.linalg.norm((point_over_time[:,5],
                                        point_over_time[:,6],
                                        point_over_time[:,7]), axis=0))


fft_freq = np.fft.fftfreq(len(point_over_time), time_step)  # Frequency bins
# Plot
plt.plot(fft_freq[:len(fft_freq)//2], np.abs(fft_output[:len(fft_output)//2]) )
plt.legend()
plt.xlabel("Frequenzy [Hz]")
plt.ylabel("Amplitude")
plt.savefig(target + '/' + f'force_single_spatial_point_fft.png', dpi=200)
plt.show()


############################################################################
print('\n--- FFT : all spatial point over time dimension ---\n')

nr_points = int(len(data)/len(t_redund))

# # loop over all spatial points in the first timestep
# for t in range(len(t_redund)):
#     bool_arr = data[:,2:5]==data[t,2:5]
#     selector = np.all(bool_arr, axis=1)

#     point_over_time = data[np.all(bool_arr, axis=1)]
#     # fft_output = np.fft.fft(point_over_time[:,5]   )
#     fft_output = np.fft.fft(np.linalg.norm((point_over_time[:,5],
#                                             point_over_time[:,6],
#                                             point_over_time[:,7]), axis=0))


#     fft_freq = np.fft.fftfreq(len(point_over_time), time_step)  # Frequency bins
#     # Plot
#     plt.plot(fft_freq[:len(fft_freq)//2], np.abs(fft_output[:len(fft_output)//2]) )

# plt.legend()
# plt.xlabel("Frequenzy [Hz]")
# plt.ylabel("Amplitude")
# plt.savefig(target + '/' + f'force_multiple_spatial_point_fft.png', dpi=200)
# plt.show()








############################################################################
### 2d fft
############################################################################
# force_mag = np.linalg.norm((data[:,5], 
#                             data[:,6],
#                             data[:,7]), axis=0)
# s = np.reshape(force_mag, (int(len(force_mag)/len(t_redund)), int(len(t_redund))) )


# # Perform 2D FFT
# fft_result = np.fft.fft2(s)

# # Shift the zero frequency component to the center
# fft_result_shifted = np.fft.fftshift(fft_result)

# # Plot the magnitude of the result
# plt.imshow(np.log(np.abs(fft_result_shifted) + 1), cmap='gray')
# plt.colorbar()
# plt.title('2D FFT Result (Magnitude)')
# plt.show()



# ############################################################################
# ### spectrogram : spatial fft, multiple ffts over time
# ############################################################################
# print('\n--- spectrogram : spatial fft, multiple ffts over time ---\n')
# signal_init = data[data[:,1]==t_redund[0]]
# fft_output_init = np.fft.fft(signal_init)
# fft_freq_init = np.fft.fftfreq(len(signal_init), time_step)  # Frequency bins
    
# spectro = np.empty((len(t_redund),  len( np.abs(fft_output_init[:len(fft_output_init)//2]) ) ))

# j = 0
# for step in t_redund:
#     data_step = data[data[:,1]==step]

#     # magnitude
#     signal = np.linalg.norm((data_step[:,5],data_step[:,6],data_step[:,7]), axis=0)
    
#     fft_output = np.fft.fft(signal)
#     fft_freq = np.fft.fftfreq(len(signal), time_step)  # Frequency bins
    
#     spectro[j,:] = np.abs(fft_output[:len(fft_output)//2])
#     j += 1
    
    
# # original frequencies for x-axis
# freq_plot = fft_freq[:len(fft_freq)//2]

# plt.contourf(spectro.T, vmin=np.min(spectro),vmax=np.max(spectro), levels=20)
# plt.show()