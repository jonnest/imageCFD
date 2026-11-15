import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as scp


flow_error = [
                -8.7,
                -41.5 ,
                60.7 ,
                -49.7 ,
                -89.7 ,
                -77.8 ,
                -46.5 ,
                -51.0 ,
                54.7 ,
                14.7 ,
                9.7 ,
             ] 

flow_error = np.array(flow_error) / 100

abs_mean = np.mean(abs(flow_error))