"""
Small routine dedicated to physionet measurements readings from ptbdb database
Database and informations available here : http://physionet.org/physiobank/database/ptbdb/
"""
__author__ =    "ThePolyscope"
__email__ =     "ThePolyscope@ThePolyscope.com"
__version__ =   "1.0.0"
__copyright__ = "Copyright 2015, ThePolyscope.com"
__license__ =   "MIT License"


import numpy
#from pylab import plot, show
#from matplotlib.pylab import plot, show
from matplotlib import pylab as pl
# Define suffix of files to parse
prefix = "patient001/s0014lre"
# Open header file in read mode
#headerFile = open("ptbdb\\" + prefix + ".hea","r")
headerFile = open("../ptbdb/" + prefix + ".hea","r")
# Open binary file in read mode
datFile = open("../ptbdb//" + prefix + ".dat","rb")

# Retrieve signal length
signalLength = int(headerFile.readline().split()[3])

# Loop over all 12 measurements from dat file
data = numpy.zeros((12,signalLength))
for sampleIdx in range(signalLength):
    # Read 12 signal from dat file
    for varIdx in range(12):
        myBytes = datFile.read(2)
        # Read binary data, rescale it and store it
        # Applicable scale factors found there : http://physionet.org/physiobank/database/ptbdb/
        data[varIdx, sampleIdx] = int.from_bytes(myBytes, byteorder='little', signed=True)/2000.

# Close ressources
headerFile.close()
datFile.close()

# Plot first measurement available
pl.ion()
pl.plot(data[0,:])
#pl.plot(data[1,:])
pl.show()
