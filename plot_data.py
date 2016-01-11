import numpy
from matplotlib import pylab as pl
from scipy import signal

import database

def remove_drift(inputData):
    """
    Remove baseline drift from inputData.
    """
    Wn = 0.001
    b, a = signal.butter(3, Wn, 'lowpass', analog=False)
    baseline = signal.filtfilt(b, a, inputData)
    outputData = inputData - baseline
    return outputData

# Get data from DB file.
prefix = "../ptbdb/"
#patientPath = "patient001/s0014lre"
patientPath = "patient001/s0010_re"
db = database.DataBase(prefix, patientPath)
data = db.get_data()

lead1_raw = data[0,:]
lead1_driftless = remove_drift(lead1_raw)

# Plot
pl.close()
pl.ion()
pl.plot(lead1_driftless)
pl.show()
