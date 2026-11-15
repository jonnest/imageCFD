import sys,os,re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as scp
from frequencyOperators import sonification

### data folder
w_dir = '/Users/jonathan/Documents/applications/cfd/cerebral/t240209/'
# w_dir = '/Users/jonathan/Documents/applications/cfd/cerebral/t240213/'

foam_folder = 't240209_vl_u3'
# foam_folder = 't240213_v11_u25'

save = True
############################################################################
### !!!
### check for spline split, probes might contain centerline points of multiple splines
# # index point of interest in spline
# t240222 - spline 0 : 800/1000
### loop over all spatial points in the first timestep
### Time    Probe1     Probe2     ...
###  X    (u1,u2,u3) (u1,u2,u3)
index_point = 0
nr_p = 250 # 137, 250
### !!!
# cut off instable fluctuations in the beginning, in seconds, maxCut 0==no cropping
min_cut_off_time = 0
max_cut_off_time = 6.6 # 6.638
# Desired cutoff frequency of the filter, Hz
# default = 20
cutoff = 20

probe_file = 'p' # U, p, force.dat
############################################################################

target = w_dir + foam_folder
probe_type = 'probes' # forces or probes 

# number of components to analyze
if probe_file == 'U':
    nr_comps = 3
    y_label = "Velocity [m/s]"
elif probe_file == 'p':
    nr_comps = 1
    y_label = "Pressure [Pa]"
    
# automatic time detection
times = os.listdir(f'{target}/postProcessing/{probe_type}')
times = ['0','1','2'    ,'3.98']

probe_paths = []
for time in times:
    probe_paths.append(f'{target}/postProcessing/{probe_type}/{time}/{probe_file}')

# read data as string array
data_arrays = []
for probe_path in probe_paths:
    data = np.genfromtxt(probe_path, dtype='str')
    # Convert elements to float
    data = np.array([[float(val.strip('()')) for val in row] for row in data])
    data_arrays.append(data)

# stack time bins of probing output
if len(data_arrays) == 1:
    data = data_arrays[0]
else:
    data = data_arrays[0]
    for i in range(1,len(data_arrays)):
        data = np.vstack((data, data_arrays[i]))

### crop unsteadiness in the beginning
# skip the first cycle
data = data[data[:, 0] > min_cut_off_time]
if max_cut_off_time != 0:
    data = data[data[:, 0] < max_cut_off_time]


print('Time step  0  : ', data[0,0])
print('Time step -1  : ', data[-1,0])


############################################################################
### averaging velocity probes in specific index area
############################################################################
x_all = np.empty((data.shape[0],nr_p))

i = 0

if probe_file == 'U':
    # d = number of point in list
    for d in np.linspace(index_point,index_point+nr_p-1,nr_p):
        # index of the point
        p = int(d * nr_comps + 1 ) 
        
        x_mag =  np.linalg.norm((data[:,p  ],
                                data[:,p+1],
                                data[:,p+2]), axis=0)
        # plt.plot(x_mag) #, label=f'{p}')
        x_all[:,i] = x_mag
        i += 1 
           
elif probe_file == 'p':
    # just skip the time column
    x_all = data[:,1:]

x_avg = np.mean(x_all, axis=1)
plt.plot(x_avg)

plt.legend()
plt.xlabel("Time [s]")
plt.ylabel(y_label)
plt.title(f'Velocity centerline probes\nIndices {int(index_point)}_{int(nr_p+index_point)}')
if save==True:
    plt.savefig(target + '/' + f'{probe_file}__avg_{int(index_point)}_{int(nr_p+index_point)}.png', dpi=200)
plt.show()

############################################################################
### fft on velocity average
############################################################################
time_step = data[0,0]-data[1,0]

fft_output =  np.fft.fft(x_avg)
fft_freq = np.fft.fftfreq(len(x_avg), time_step)  # Frequency bins
# Plot
plt.plot(np.abs(fft_freq[:len(fft_freq)//2]), np.abs(fft_output[:len(fft_output)//2]) )

plt.legend()
plt.xlabel("Frequenzy (Hz)")
plt.ylabel("Amplitude")
plt.yscale("log")
plt.title(f'FFT on average centerline probes\nindices {int(index_point)}_{int(nr_p+index_point)}')
if save==True:
    plt.savefig(target + '/' + f'{probe_file}_fft_x_avg_{int(index_point)}_{int(nr_p+index_point)}.png', dpi=200)
plt.show()

############################################################################
### spectogram
############################################################################
### - power spectral density 
### - display the logarithm of the power (to reduce the dynamic range)
### power spectral density : P_i = 1/(sum_weights**2) * mag**2

# sampling frequency
fs = 2500

# spectral power density - colorplot boundaries
vmax = -20
vmin = -70

# # zeropad signal to have full cycle of 1 second
# x_mag = np.pad(x_mag, (fs-len(data),0))

# number of segments - resulting from number of windows over cycle 
# ~ 36 windows / second
# n_windows = 30

### number of samples per segment
# nperseg = int(len(x_mag)/n_windows)
nperseg = int(fs/10)

# length of fft signal, zero-pad missing signal
nfft = 256
    
### power spectral matrix
_, _, Sxx_init = sonification(x_avg, fs, nperseg, nfft)
# initializaion
Sxx_all = np.zeros((Sxx_init.shape[0],Sxx_init.shape[1],nr_p))

for idx in range(nr_p):
    x_mag = x_all[:,idx]
    # sonification   
    f, t, Sxx_i = sonification(x_mag, fs, nperseg, nfft, cutoff=cutoff)
    Sxx_all[...,idx] = Sxx_i

# averaging over all spectrograms
Sxx = np.mean(Sxx_all, axis=2)
 
# plot power spectral density logrithmic
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', vmin=vmin, vmax=vmax)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title(f'Spectrogram centerline probes\nIndices {int(index_point)}_{int(nr_p+index_point)}')
plt.colorbar(label='Intensity [dB]')

spectro_name = f'{probe_file}_spectrogram_log_avg{int(index_point)}_{int(nr_p+index_point)}_cut{cutoff}.png'
if save==True:
    plt.savefig(target + '/' + spectro_name, dpi=200)
plt.show()
