import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def linear(x: np.ndarray, fit: np.ndarray):
    """
    Applies a simple y = mx + b transformation of a single value or array-like input.
    The second parameter (fit) should be an array of the coefficients as returned by 
    the NumPy function polyfit(deg=1).
    """
    return (x * fit[0]) + fit[1] 

def quadratic(x: np.ndarray, fit: np.ndarray):
    """
    Applies a simple y = ax^2 + bx + c transformation of a single value or array-like input.
    The second parameter (fit) should be an array of the coefficients as returned by 
    the NumPy function polyfit(deg=2).
    """
    return (x * x * fit[0]) + (x * fit[1]) + fit[2]

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

# *********************
# ** ALGORITHM SETUP **
# *********************
PLOT_INTERMEDIATE = False   # controls whether each intermediate plot is displayed as the window slides
WIN_WIDTH = 5               # width of window, sets the number of pts to include in each fit
DEV_CUTOFF = 0.1            # if slope has a larger deviation than this between two consecutive windows, the region is considered as a new domain

shifts = [x for x in range(len(x_data) - WIN_WIDTH + 1)]
slopes_Q = []
slopes_f = []

# ************************
# ** SLIDING WINDOW ALG **
# ************************
for shift in shifts:
    # window end (exclusive idx)
    WIN_START = shift
    WIN_END = WIN_START + WIN_WIDTH
    
    # PERFORM FIT, FIRST DEGREE, OBTAIN SLOPE AND INTERCEPT
    fit_Q = np.polyfit(x_data[WIN_START:WIN_END], y_data_Q[WIN_START:WIN_END], deg=1)
    fit_f = np.polyfit(x_data[WIN_START:WIN_END], y_data_f[WIN_START:WIN_END], deg=1)
    y_fit_Q = linear(x_data, fit_Q)
    y_fit_f = linear(x_data, fit_f)

    print("SLOPE:", fit_Q[0])
    slopes_Q.append(fit_Q[0])
    slopes_f.append(fit_f[0])

    # PLOT 
    if PLOT_INTERMEDIATE:
        fig, axs = plt.subplots(2, 1)
        fig.suptitle("Shifts vs. Insertion Height")
        axs[0].plot(x_data, y_data_Q, '+')
        axs[0].plot(x_data[WIN_START:WIN_END], y_fit_Q[WIN_START:WIN_END], 'g-')
        axs[0].set(ylabel="Q Shift")
        axs[1].plot(x_data, y_data_f, '+')
        axs[1].plot(x_data[WIN_START:WIN_END], y_fit_f[WIN_START:WIN_END], 'g-')
        axs[1].set(ylabel="Freq Shift")
        plt.show()

# this list will store the domain id for each "shift" value of the sliding window
# consecutive shift values corresponding to similar slopes will be labeled with the same id, meaning they are in the same domain
currentDomain_Q = 0                 # initialize domain
currentDomain_f = 0                 
currentSlope_Q = slopes_Q[0]        # initialize slope
currentSlope_f = slopes_f[0]    
domain_ids_Q = [currentDomain_Q]    # intialize list of ids
domain_ids_f = [currentDomain_f]   


for i in range(1, len(shifts)):
    slope_Q_dev = abs(slopes_Q[i] - currentSlope_Q) / currentSlope_Q   # fractional deviation between the most recently selected slope and the current one in the list 
    slope_f_dev = abs(slopes_f[i] - currentSlope_f) / currentSlope_f   

    if slope_Q_dev < DEV_CUTOFF:   # no significant deviation for new slope, assign to current domain
        domain_ids_Q.append(currentDomain_Q)
    else:                   # significant deviation for new slope, increment domain id and add to list, also reassign new slope
        currentDomain_Q += 1
        domain_ids_Q.append(currentDomain_Q)
        currentSlope_Q = slopes_Q[i]

    if slope_f_dev < DEV_CUTOFF:   #
        domain_ids_f.append(currentDomain_f)
    else:                   
        currentDomain_f += 1
        domain_ids_f.append(currentDomain_f)
        currentSlope_f = slopes_f[i]
    
print(domain_ids_Q, len(domain_ids_Q), len(shifts))
print(domain_ids_f, len(domain_ids_f), len(shifts))

# plot slope against window shift. Linear domains will look like a flat line
fig, axs = plt.subplots(2, 1)
fig.suptitle("Window Shift vs. Slope")
axs[0].scatter(shifts, slopes_Q, c=domain_ids_Q)
axs[0].set(title="Q Shift")
axs[1].scatter(shifts, slopes_f, c=domain_ids_f)
axs[1].set(title="Freq Shift")
plt.show()















