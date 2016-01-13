import numpy as np
from matplotlib import pyplot as plt

from database import DataBase
from dataconditioner import DataConditioner

prefix = "../ptbdb/"
db = DataBase(prefix)
dc = DataConditioner()

dataHealthy = list()
dataMI = list()
with open(prefix + "RECORDS") as f:
    for line in f:
        patientPath = line.rstrip('\n')
        ## Get diagnosis (pathologic or not).
        print(patientPath)
        diagnosis = db.get_diagnosis(patientPath)
        if diagnosis == 'Healthy control':
            lead1_raw = db.get_data(patientPath, lead = 0)
            slices = dc.condition_signal(lead1_raw)
            dataHealthy += slices
        elif diagnosis == 'Myocardial infarction':
            lead1_raw = db.get_data(patientPath, lead = 0)
            slices = dc.condition_signal(lead1_raw)
            dataMI += slices
        else :
            print(diagnosis)
            continue

lengths = list()
for arr in dataHealthy:
    lengths.append(len(arr))
lengths = np.array(lengths)

# Plot
plt.close()
plt.ion()
for sample in dataHealthy:
    plt.plot(sample, 'b')
for sample in dataMI:
    plt.plot(sample, 'r')
plt.show()
