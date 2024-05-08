import sys,os,re
import numpy as np
sys.path.append('/Users/jonathan/Documents/code/pytilities/utilities')
from mytools import *
import pandas as pd
import matplotlib.pyplot as plt
import csv


### data folder
w_dir = '/Users/jonathan/Documents/data/deepFlow/segmentations/'
folder = 'polar_tinnitus_240110'
probe_file = 'spline_2.vtk'



target = w_dir + folder
file_path = os.path.join(target,probe_file)

vtk_data = pv.read(file_path)
data = vtk_data.points

# sample data, crop sample points
print('\nLength of data  ', len(data))

# factor to crop, every n-th element
f_crop = 4

data = data[::f_crop,:]
print('\nLength of cropped data  ', len(data), '\n')

for i in range(len(data)):
    # Join the numbers with tabs and add brackets at the beginning and end
    formatted_row = '(' + str(data[i,0]) + '\t' + str(data[i,1]) + '\t' + str(data[i,2])+ ')'
    print(formatted_row)
    