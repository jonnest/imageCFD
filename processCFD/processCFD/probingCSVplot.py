import sys,os,re
import numpy as np
sys.path.append('/Users/jonathan/Documents/code/pytilities/utilities')
from mytools import *
import pandas as pd
import matplotlib.pyplot as plt

### data folder
w_dir = '/Users/jonathan/Documents/applications/cfd/cerebral_veins/'
foam_folder = 'tinnitus_240110_seg1_les_m1'

target = w_dir + foam_folder
probe_file = 'probe_bulge_point.csv'


file_path = os.path.join(target,probe_file)

df = pd.read_csv(file_path)

t = np.linspace(0,2,len(df))

# Define a function to compute L2 norm row-wise
def compute_row_norm(row):
    return np.linalg.norm(row)


# Apply the function to each row of the DataFrame
row_norms = df.loc[:,['U:0', 'U:1', 'U:2']].apply(compute_row_norm, axis=1)

# Add row-wise L2 norm as a new column
df['U_mag'] = row_norms


plt.plot(t, df['U:0'], label='ux')
plt.plot(t, df['U:1'], label='uy')
plt.plot(t, df['U:2'], label='uz')
plt.plot(t, df['U_mag'], label='umag')
plt.legend()
plt.show()


### Perform FFT - velocity and pressure
signals = [df['U:0'],df['U:1'],df['U:2'], df['p'] ]
plt.figure(figsize=(10, 5))
for signal in signals:
    fft_output = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), 1/len(signal) )  # Frequency bins
    # Plot
    plt.plot(fft_freq[:len(fft_freq)//2], np.abs(fft_output[:len(fft_output)//2]), label=f'{signal.name}')  # Plot only positive frequencies

plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.xlim([0,20])
plt.title('FFT of the Signal')
plt.savefig(target+'/fft_probe_up.png', dpi=200)
plt.show()



### Perform FFT - vorticity
signals = [df['Vorticity:0'], df['Vorticity:1'], df['Vorticity:2']] 
plt.figure(figsize=(10, 5))
for signal in signals:
    fft_output = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), 1/len(signal) )  # Frequency bins
    # Plot
    plt.plot(fft_freq[:len(fft_freq)//2], np.abs(fft_output[:len(fft_output)//2]), label=f'{signal.name}')  # Plot only positive frequencies

plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.xlim([0,20])
plt.title('FFT of the Signal')
plt.savefig(target+'/fft_probe_w.png', dpi=200)
plt.show()


### Perform FFT - rest
signals = [df['turbulenceProperties:k']] 
plt.figure(figsize=(10, 5))
for signal in signals:
    fft_output = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), 1/len(signal) )  # Frequency bins
    # Plot
    plt.plot(fft_freq[:len(fft_freq)//2], np.abs(fft_output[:len(fft_output)//2]), label=f'{signal.name}')  # Plot only positive frequencies

plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.xlim([0,20])
plt.title('FFT of the Signal')
plt.savefig(target+'/fft_probe_k.png', dpi=200)
plt.show()