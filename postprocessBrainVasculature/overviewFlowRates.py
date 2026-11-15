import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


### data folder
w_dir = '/Users/jonathan/Documents/data/deepFlow/segmentations/'


df = pd.read_csv(w_dir + '/flowrates.csv')

plt.figure()
sns.boxplot(x='vessel', y='inflow', data=df)

plt.figure()
sns.boxplot(x='vessel', y='outflow', data=df)


### create new data structure for boxplot
inflow = df.drop(columns=['outflow', 'error'])
inflow = inflow.rename(columns={"inflow": "flowrate"})
inflow['flow'] = 0

outflow = df.drop(columns=['inflow', 'error'])
outflow = outflow.rename(columns={"outflow": "flowrate"})
outflow['flow'] = 1

flow = pd.concat([inflow, outflow], ignore_index=True,axis=0)
plt.figure()
sns.boxplot(x='vessel', y='flowrate', hue='flow', data=flow)
