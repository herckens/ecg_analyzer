import numpy as np
import linecache

class DataBase:
    """
    Small routine dedicated to physionet measurements readings from ptbdb database
    Database and informations available here : http://physionet.org/physiobank/database/ptbdb/
    """
    __author__ =    "ThePolyscope"
    __email__ =     "ThePolyscope@ThePolyscope.com"
    __version__ =   "1.0.0"
    __copyright__ = "Copyright 2015, ThePolyscope.com"
    __license__ =   "MIT License"

    def __init__(self, prefix):
        self.prefix = prefix

    def get_data(self, patientPath, lead = None):
        """
        Read the raw ECG signal from the binary file.
        If lead is None:
            Return a numpy.array containing all 12 arrays with the signals.
        If lead is an integer between 0 and 11
            Return one numpy.array with the signal of specified lead.
        """
        # Open header file in read mode
        headerFile = open(self.prefix + patientPath + ".hea","r")
        # Open binary file in read mode
        datFile = open(self.prefix + patientPath + ".dat","rb")

        # Retrieve signal length
        signalLength = int(headerFile.readline().split()[3])

        if lead is not None:
            # Extract measurement number lead from dat file
            data = np.zeros((signalLength))
            for sampleIdx in range(signalLength):
                for varIdx in range(12):
                    myBytes = datFile.read(2)
                    if varIdx == lead:
                        # Read binary data, rescale it and store it
                        # Applicable scale factors found there : http://physionet.org/physiobank/database/ptbdb/
                        data[sampleIdx] = int.from_bytes(myBytes, byteorder='little', signed=True)/2000.
        else:
            # Loop over all 12 measurements from dat file
            data = np.zeros((12,signalLength))
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

    def get_diagnosis(self, patientPath):
        """
        Get the patient's diagnosis from the header file.
        """
        line = linecache.getline(self.prefix + patientPath + ".hea", 23)
        diagnosis = ' '.join(line.split()[4:])
        return diagnosis

    def has_myocardial_infarction(self, patientPath):
        """
        Returns true, if this patient is diagnosed with myocardial infarction.
        Returns false, in all other cases.
        """
        diagnosis = self.get_diagnosis(patientPath)
        if diagnosis == 'Myocardial infarction':
            return True
        else :
            return False
