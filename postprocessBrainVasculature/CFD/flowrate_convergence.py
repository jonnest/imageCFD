import sys,os,re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv



### data folder
w_dir = '/Users/jonathan/Documents/applications/cfd/eccomas_case/mesh_study'


foam_folders = [#"/foam_case_v1_m1",
                "/foam_case_v1_m2",
                #"/foam_case_v1_m3",
                "/foam_case_v1_m4",
                #"/foam_case_v1_m5",
                ]

labels = [#"m1",
          "m2",
          #"m3",
          "m4",
          #"m5"
          ]

# outflow 0, 3
probe_file = 'bnd1.csv'

# inflow = [1,2]
# outflow = [0,3,4]

nr_cells = [
            #445285,
            769827,
            #1285970,
            3261048,
            #4839033
            ]

# name_input = probe_file.split('.')[0]
name_input = probe_file.split('.')[0] + '_solver'

# conservation_name = 'm3'

targets = []
for foam_folder in foam_folders:
    targets += [w_dir + foam_folder]

file_paths = []
for target in targets:
    file_paths += [ os.path.join(target,probe_file) ]

data_arrs = []
for file_path in file_paths:
    # data_arrs += [ np.loadtxt(file_path, skiprows=1, delimiter=',') ]
    data_arrs += [ pd.read_csv(file_path) ]


############################################################################
### Richardson Extrapolation
############################################################################

# pruning to same range of simulation time
t_min = 1
t_max = 3

data_arrs_cut = []
for arr in data_arrs:
    tmp = arr[arr['Time'].between(t_min, t_max)]
    data_arrs_cut.append(tmp)

# Step sizes (assuming uniform grid refinement)
hi = []
for i in range(len(nr_cells)):
    hi.append( 1 / nr_cells[i]**(1/3))

### Calculate the apparent order of convergence p, R is result of some metric
### e.g :  p = (np.log((R_2_4m - R_1_5m) / (R_1_5m - R_500k))) / np.log(h3 / h2)

'''
# Calculate the apparent order of convergence p
f0 = np.array(data_arrs_cut[0]["magU"])
f1 = np.array(data_arrs_cut[1]["magU"])
f2 = np.array(data_arrs_cut[2]["magU"])

# Calculate the apparent order of convergence p
p = (np.log(abs((f0 - f1) / (f2 - f1)))) / np.log(hi[0] / hi[1])

# Richardson extrapolation to infinite cells
# R_inf = f0 + (f0 - f1) / ((hi[0] / hi[1])**p - 1)

R_inf = f0 + (f0 * hi[1]**2 - f1 * hi[0]**2) / (hi[1]**2 - hi[0]**2) 
'''
############################################################################
### 1 spatial point over time dimension
############################################################################
print('\n--- Flowrate over time dimension ---\n')

# # bnd integral over patch header
# header = ["TimeStep","Time","U:0","U:1","U:2","forces:force:0","forces:force:1",
# "forces:force:2","forces:moment:0","forces:moment:1","forces:moment:2","magU",
# "nut","p","Points:0","Points:1","Points:2"]

# hdr_idx = header.index("magU")
# input = data[:,hdr_idx]

for i in range(len(data_arrs)):
    data = data_arrs[i]
    
    condition = data['Time'] <= 2
    data = data.loc[condition]
    
    t = data["Time"]
    # input = data["magU"]
    input = np.linalg.norm([data["U:0"],data["U:1"],data["U:2"]], axis=0)
    
    # Plot
    plt.plot(t, input, 'o', label=labels[i])

### plot Richardson extrapolation
# plt.plot(data_arrs_cut[0]['Time'], R_inf*1e6, '.', label='Richardson')

plt.legend()
plt.title(f"Flowrate of outlet boundary : {name_input}")
plt.xlabel("Time (s)")
plt.ylabel("Flowrate (ml/s)")
plt.savefig(w_dir + '/' + f'{name_input}_flowrate.png', dpi=200)
plt.show()


############################################################################
### error over time
############################################################################
# print('\n--- Flowrate error ---\n')

# t_min = 1
# t_max = 3

# data_hq = data_arrs[-1]
# data_hq = data_hq[data_hq['Time'].between(t_min, t_max)]


# for i in range(len(data_arrs)-1):
#     data = data_arrs[i]
#     data = data[data['Time'].between(t_min, t_max)]
    
#     t = data["Time"]
    
#     err_per = ( np.array(data_hq["magU"]) - np.array(data["magU"]) ) / np.array(data_hq["magU"]) * 100
    
#     print("data hq", data_hq["magU"].shape)
#     print("data ", data["magU"].shape)
    
#     print("shape error ", err_per.shape)
    
#     # Plot
#     plt.plot(t, err_per, 'o', label=labels[i])

# plt.legend()
# plt.title(f"Flowrate of outlet boundary : {name_input}")
# plt.xlabel("Time (s)")
# plt.ylabel("Flowrate error (%)")
# plt.savefig(w_dir + '/' + f'{name_input}_flowrate_error.png', dpi=200)
# plt.show()


############################################################################
# mass conservation of open foam CFD simulation
############################################################################
'''
foam_folder = foam_folders[1]


target = w_dir + foam_folder

inflow_paths = []
for i in inflow:
    inflow_paths += [ os.path.join(target,f'bnd{i}.csv') ]

outflow_paths = []
for i in outflow:
    outflow_paths += [ os.path.join(target,f'bnd{i}.csv') ]
    
in_arrs = []
for inflow_path in inflow_paths:
    in_arrs += [ pd.read_csv(inflow_path) ]

out_arrs = []
for outflow_path in outflow_paths:
    out_arrs += [ pd.read_csv(outflow_path) ]


inflow_rate = []
for i, j in enumerate(inflow):
    data = in_arrs[i]
    
    t = data["Time"]
    input = data["magU"]
    inflow_rate += [np.array(input)]
    
outflow_rate = []
for i, j in enumerate(outflow):
    print(i, j)
    data = out_arrs[i]
    
    t = data["Time"]
    output = data["magU"]
    outflow_rate += [np.array(output)]

# plot
inflow_sum = sum(inflow_rate)
outflow_sum = sum(outflow_rate)

plt.plot(t, inflow_sum, label='inflow')
plt.plot(t, outflow_sum, label='outflow')

plt.legend()
plt.title(f"Mass conservation")
plt.xlabel("Time (s)")
plt.ylabel("Flowrate (ml/s)")
plt.savefig(w_dir + '/' + f'{conservation_name}_flowrate_conservation.png', dpi=200)
plt.show()
'''