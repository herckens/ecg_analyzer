import numpy as np
#from matplotlib import pylab as pl
from matplotlib import pyplot as plt
from scipy import signal

from database import DataBase
from dataconditioner import DataConditioner


# Get data from DB file.
prefix = "../ptbdb/"
#patientPath = "patient001/s0014lre"
patientPath = "patient001/s0010_re"
db = DataBase(prefix, patientPath)
lead1_raw = db.get_data(lead = 0)

dc = DataConditioner()
# Remove baseline drift.
lead1_driftless = dc.remove_drift(lead1_raw)
# Smooth the data.
Wn = 0.08
b, a = signal.butter(6, Wn, 'lowpass', analog=False)
lead1_smoothed = signal.filtfilt(b, a, lead1_driftless)
# Find location and value of R peaks.
rPeaksInd, rPeaksVal = dc.find_r_peaks(lead1_smoothed)

# Plot
plt.close()
plt.ion()
#plt.plot(lead1_driftless)
plt.plot(lead1_smoothed)
plt.scatter(rPeaksInd, rPeaksVal)
plt.show()
