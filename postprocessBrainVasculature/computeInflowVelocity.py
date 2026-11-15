''' Inflow velocities for sUbend

'''
import numpy as np

# Parameters
nu = 3e-6 

a1 = 0.000771693
a2 = 0.000169988

d1 = np.sqrt(a1/np.pi)*2
d2 = np.sqrt(a2/np.pi)*2


Re_list = [50,100,200,300,400,500]

# Get Inflow Velocity

for Re in Re_list:
    u_inflow = Re*nu/d1
    print("\nRe_inflow     ", Re)
    print("U_inflow      ", round(u_inflow,7))

    u2 = u_inflow * a1/a2
    Re2 = u2*d2/nu
    print("Re_stenosis   ", round(Re2,1))
    