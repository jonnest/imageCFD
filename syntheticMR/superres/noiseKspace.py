''' How to introduce Gaussian noise in k-space
'''

import numpy as np
import matplotlib.pyplot as plt

from fourierOperators import * 
from gridOperators import interpolate_3d_grid, downsample_kspace_add_complex_noise


norm = 'ortho'

# Create grid indices
x = np.linspace(0, 2*np.pi, 120)
y = np.linspace(0, 2*np.pi, 90)
z = np.linspace(0, 2*np.pi, 30)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Real part: smooth variation using a sine wave
real_part = np.sin(X) + 2*np.cos(Y) + 3*np.sin(Z)

# Imaginary part: another smooth variation (e.g., cosine wave)
imaginary_part = np.cos(X) - np.sin(Y) + np.cos(Z)

# Combine real and imaginary parts into a complex array
S = real_part + 1j * imaginary_part

snr = 20

# normalize the fourier transform - then gaussian transforms to gaussian
Sf = np.fft.fftn(S, norm='ortho')
noise = np.random.normal(0,np.mean(np.abs(S))/snr,S.shape) + \
     1j*np.random.normal(0,np.mean(np.abs(S))/snr,S.shape)
     
S_noisy = np.fft.ifftn(Sf + noise, norm='ortho')

print('Target SNR:              ', str(snr))
print('Estimated SNR:           ', np.mean(np.abs(S_noisy)) / np.std(np.abs(S)-np.abs(S_noisy)))
print('Wrong SNR computation:   ', np.mean(np.abs(S)) / np.std(np.abs(S_noisy)))

# downsample
complex_fft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(S), norm=norm))

# take mean of magnitude of original HR large image, use size of downsampled image
noise = np.random.normal(0,np.mean(np.abs(S))/snr, complex_fft.shape) + \
     1j*np.random.normal(0,np.mean(np.abs(S))/snr, complex_fft.shape)
     
complex_img_noise = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(complex_fft + noise), norm=norm))

complex_low = interpolate_3d_grid(complex_img_noise, 0.5, method='fft',
                              axes=(0,1,2),kwindow='hybrid',
                              alpha=0.99)


snr_lr = np.mean(np.abs(complex_img_noise))/np.std(np.abs(complex_img_noise)-np.abs(noise))

snr_hr = np.mean(np.abs(complex_img_noise))/np.std(np.abs(complex_img_noise))

print("\nsnr low res  ", snr_lr)
print("snr high res  ", snr_hr)


# Plot a slice of the real part of S and S_noisy side by side
slice_idx = 12  # Take a slice along the z-axis

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot original real part of S (slice)
im1 = axes[0].imshow(np.abs(S[:, :, slice_idx]), cmap='viridis', vmin=0, vmax=np.abs(S[:, :, slice_idx]).max())
axes[0].set_title('Original S (magnitude)')
plt.colorbar(im1, ax=axes[0])

# Plot real part of S_noisy (slice)
im2 = axes[1].imshow(np.abs(S_noisy[:, :, slice_idx]), cmap='viridis', vmin=0, vmax=np.abs(S[:, :, slice_idx]).max())
axes[1].set_title('Noisy S (magnitude)')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()