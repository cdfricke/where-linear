import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

cwd = os.getcwd()
print("Current Working Directory:", cwd)

fileToRead = "data/090224/RexoliteRod1_PostBake.csv"

print("Reading:", fileToRead)
df = pd.read_csv(fileToRead)

# get f0 and Q0
init_freq = df['Frequency (GHz)'].iloc[0]
init_Q = df['Q'].iloc[0]

# create columns for freq_shift and Q_shift
df['freq_shift'] = (init_freq - df['Frequency (GHz)']) / init_freq
df['Q_shift'] = (1.0 / df['Q']) - (1.0 / init_Q) 

# convert to numpy arrays
x_data = df['height (mm)'].to_numpy()
y_data0 = df['Q_shift'].to_numpy()
y_data1 = df['freq_shift'].to_numpy()

# PLOT
fig, axs = plt.subplots(2, 1)
fig.suptitle("Shifts vs. Insertion Height")
axs[0].plot(x_data, y_data0, '+')
axs[0].set(ylabel="Q Shift")
axs[1].plot(x_data, y_data1, '+')
axs[1].set(ylabel="Freq Shift")
plt.show()







