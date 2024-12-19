# Programmer: Connor Fricke
# File: main.py
# Latest Rev: 19-Dec-2024
# Desc: main script for testing where_linear module on available
# datasets for a comparison of Q shift and frequency shift vs. insertion height.

import pandas as pd
import where_linear as wl
import os

# for debugging
cwd = os.getcwd()
print("Current Working Directory:", cwd)

# paths of various files (relative to project directory)
filesToRead = ["data/090224/Empty_Vial7_PostBake.csv", 
               "data/090224/Empty_Vial7_USD_PostBake.csv", 
               "data/090224/RexoliteRod1_PostBake.csv", 
               "data/090224/TeflonRod1_PostBake.csv",
               "data/090224/Vial7_PostBake_Teflon1.csv", 
               "data/090224/Vial7_Rexolite3_PostBake.csv"
               ]

# ******************
# ** ACQUIRE DATA **
# ******************
print("Reading:", filesToRead[0])
df = pd.read_csv(filesToRead[0])

# get f0 and Q0
init_freq = df['Frequency (GHz)'].iloc[0]
init_Q = df['Q'].iloc[0]

# create columns for freq_shift and Q_shift
df['freq_shift'] = (init_freq - df['Frequency (GHz)']) / init_freq
df['Q_shift'] = (1.0 / df['Q']) - (1.0 / init_Q) 

# convert to numpy arrays
x_data = df['height (mm)'].to_numpy()
y_data_Q = df['Q_shift'].to_numpy()
y_data_f = df['freq_shift'].to_numpy()

LDF = wl.LinearDomainFinder()
LDF.setX(x_data, label='height (mm)')
LDF.setY(y_data_Q, label='Q_shift')
LDF.setVerbosity(1)
LDF.slidingWindowFind(WIN_SIZE=5, FDEV_CUT=0.1)

resultDomain = LDF.LLD
resultSlope = LDF.slope
















