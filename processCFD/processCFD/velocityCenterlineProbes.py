import sys,os,re
import numpy as np
sys.path.append('/Users/jonathan/Documents/code/pytilities/utilities')
from mytools import *
import pandas as pd
import matplotlib.pyplot as plt

### data folder
w_dir = '/Users/jonathan/Documents/applications/cfd/cerebral_veins/'

# foam_folder = 'tinnitus_240110_seg1_les_m05_dt'
foam_folder = 'tinnitus_240110_s2_m08'


times = ['0', '1.01', '1.7', '2']

target = w_dir + foam_folder


probe_file = 'U' # U, p, force.dat
probe_type = 'probes' # forces or probes 

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
data = np.vstack((data_arrays[0], data_arrays[1]))

index_point = 800
start = index_point * 3 + 1
nr_p = 30
for p in np.arange(start,start+nr_p*3,3).astype(int):
    print(p)
    u_mag =  np.linalg.norm((data[:,p+0],
                             data[:,p+1],
                             data[:,p+2]), axis=0)
    plt.plot(u_mag) #, label=f'{p}')
    
plt.legend()
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

############################################################################
print('\n--- FFT : all spatial point over time dimension ---\n')

time_step = data[0,0]-data[1,0]

# # loop over all spatial points in the first timestep
index_point = 800
nr_p = 50

start = index_point * 3 + 1

for p in np.arange(start,start+nr_p*3,3).astype(int):
    fft_output =  np.fft.fft(np.linalg.norm((data[:,p+0],
                                             data[:,p+1],
                                             data[:,p+2]), axis=0))


    fft_freq = np.fft.fftfreq(len(data), time_step)  # Frequency bins
    # Plot
    plt.plot(np.abs(fft_freq[:len(fft_freq)//2]), np.abs(fft_output[:len(fft_output)//2]) )

plt.legend()
plt.xlabel("Frequenzy [Hz]")
plt.ylabel("Amplitude")
plt.yscale("log")
plt.savefig(target + '/' + f'fft_spatial_points.png', dpi=200)
plt.show()