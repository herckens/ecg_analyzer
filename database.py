import numpy

class DataBase:
    def __init__(self, prefix, patientPath):
        self.prefix = prefix
        self.patientPath = patientPath

    def get_data(self):
        """
        Small routine dedicated to physionet measurements readings from ptbdb database
        Database and informations available here : http://physionet.org/physiobank/database/ptbdb/
        """
        __author__ =    "ThePolyscope"
        __email__ =     "ThePolyscope@ThePolyscope.com"
        __version__ =   "1.0.0"
        __copyright__ = "Copyright 2015, ThePolyscope.com"
        __license__ =   "MIT License"
        # Open header file in read mode
        headerFile = open(self.prefix + self.patientPath + ".hea","r")
        # Open binary file in read mode
        datFile = open(self.prefix + self.patientPath + ".dat","rb")

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

        return data

