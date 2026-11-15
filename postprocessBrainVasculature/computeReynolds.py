import numpy as np

# Parameters
nu = 3e-6


# 


# Get Reynolds number

q1 = 6.91714e-06
a1 = 0.000771693

u1 = q1/a1

d1 = np.sqrt(a1/np.pi)*2

Re1 = u1 * d1 / nu

print("U      [m/s]  ", u1) 
print("D        [m]  ", d1) 
print("Re       [-]  ", Re1)