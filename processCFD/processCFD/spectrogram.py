import sys,os,re
import numpy as np
sys.path.append('/Users/jonathan/Documents/code/pytilities/utilities')
from mytools import *
import pandas as pd
import matplotlib.pyplot as plt
import csv


### data folder
w_dir = '/Users/jonathan/Documents/applications/cfd/cerebral_veins/'
foam_folder = 'tinnitus_240110_seg1_les_m0_analysis'

target = w_dir + foam_folder
probe_file = 'bulge_box_time.csv'


file_path = os.path.join(target,probe_file)

# df = pd.read_csv(file_path)
data = np.loadtxt(file_path, skiprows=1, delimiter=',')

### Perform FFT - velocity and pressure

t_redund = np.unique(data[:,1])
time_step = t_redund[1]-t_redund[0]