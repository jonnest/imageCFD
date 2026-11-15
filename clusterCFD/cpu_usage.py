import os,sys
import numpy as np 
import matplotlib.pyplot as plt

### INPUT
number_cells = 660000

# check cpu usage with my_jobs

### cpu usage data 
# --- nr_cell --- nr_cpus --- cpu_usage
data = np.array(
    (
        (  492573,  24, 27.48 ),      # t240209_v1_m1
        (  843525,  24, 99.07 ),      # t240209_v1_m2
        ( 1365453,  38, 98.00 ),      # t240209_v53_u3
        ( 1698883,  48, 68.17 ),      # t240209_v1_m3
        ( 4657738, 120, 24.92 ),      # t240209_v1_m4

    ))


# plot : efficiency over cells per core 

cells_per_core = data[:,0]/data[:,1]

plt.plot(cells_per_core, data[:,2])
# plt.legend()
plt.xlabel("cells per cpu")
plt.ylabel("cpu utilization")

# plt.savefig(flow_rate_folder + '/' + f'flowrate_bnd{boundary}.png', dpi=200)
plt.show()

opt_cells = cells_per_core[np.argmax(data[:,2])]
print('\nOptimum cells per core  ', opt_cells)
print('\nNumber of cores for CFD  ', int(number_cells/opt_cells))
