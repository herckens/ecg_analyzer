import numpy as np

from database import DataBase
from dataconditioner import DataConditioner

class DataSet:
    def __init__(self, name, prefix):
        self._name = name
        self._prefix = prefix
        self._recordsFile = "RECORDS_" + self._name
        self._fileData = name + "Data.npy"
        self._fileLabels = name + "Labels.npy"
        self._fileDataHealthy = name + "DataHealthy.npy"
        self._fileLabelsHealthy = name + "LabelsHealthy.npy"
        self._fileDataMI = name + "DataMI.npy"
        self._fileLabelsMI = name + "LabelsMI.npy"
        self.load_data()

    @property
    def recordsFile(self):
        return self._recordsFile
    @property
    def data(self):
        return self._data
    @property
    def labels(self):
        return self._labels
    @property
    def dataHealthy(self):
        return self._dataHealthy
    @property
    def labelsHealthy(self):
        return self._labelsHealthy
    @property
    def dataMI(self):
        return self._dataMI
    @property
    def labelsMI(self):
        return self._labelsMI

    def next_batch(self, batchSize):
        """
        Return the next `batch_size` examples from this data set.
        Returns tuple of data, labels.
        """
        data = list()
        labels = list()
        for i in range(batchSize):
            if i%2 == 0:
                ind = np.random.randint(len(self._dataHealthy))
                data.append(self._dataHealthy[ind])
                labels.append(self._labelsHealthy[ind])
            else:
                ind = np.random.randint(len(self._dataMI))
                data.append(self._dataMI[ind])
                labels.append(self._labelsMI[ind])
        return data, labels

    def shuffle_data(self, data, labels):
        length = len(data)
        indices = list(range(0, length))
        np.random.shuffle(indices)
        dataShuffled = list()
        labelsShuffled = list()
        for i in indices:
            dataShuffled.append(data[i])
            labelsShuffled.append(labels[i])
        return dataShuffled, labelsShuffled

    def import_dataset(self):
        db = DataBase(prefix)
        dc = DataConditioner()
        data = list()
        labels = list()
        dataHealthy = list()
        labelsHealthy = list()
        dataMI = list()
        labelsMI = list()
        with open(prefix + recordsFile) as f:
            for line in f:
                patientPath = line.rstrip('\n')
                print(patientPath)
                diagnosis = db.get_diagnosis(patientPath)
                if diagnosis == 'Healthy control':
                    lead1_raw = db.get_data(patientPath, lead = 0)
                    slices = dc.condition_signal(lead1_raw)
                    data += slices
                    dataHealthy += slices
                    for i in range(0, len(slices)):
                        labels.append([1,0])
                        labelsHealthy.append([1,0])
                elif diagnosis == 'Myocardial infarction':
                    lead1_raw = db.get_data(patientPath, lead = 0)
                    slices = dc.condition_signal(lead1_raw)
                    data += slices
                    dataMI += slices
                    for i in range(0, len(slices)):
                        labels.append([0,1])
                        labelsMI.append([0,1])
                else :
                    continue
        # Shuffle the data.
        self._data, self._labels= shuffle_data(data, labels)
        self._dataHealthy, self._labelsHealthy= shuffle_data(dataHealthy, labelsHealthy)
        self._dataMI, self._labelsMI= shuffle_data(dataMI, labelsMI)

    def load_data(self):
        try :
            # If previous data exists in files, load it.
            print(self._name + ": Loading previous data.")
            self._data = np.load(self._fileData)
            self._labels = np.load(self._fileLabels)
            self._dataHealthy = np.load(self._fileDataHealthy)
            self._labelsHealthy = np.load(self._fileLabelsHealthy)
            self._dataMI = np.load(self._fileDataMI)
            self._labelsMI = np.load(self._fileLabelsMI)
            print(self._name + ": Finished loading previous data.")
        except IOError:
            # If no previous files exist, reimport the ECG data.
            print(self._name + ": Reimporting data.")
            data = self.import_dataset(self._recordsFile)
            self._data = np.array(data[0])
            self._labels = np.array(data[1])
            self._dataHealthy = np.array(data[2])
            self._labelsHealthy = np.array(data[3])
            self._dataMI = np.array(data[4])
            self._labelsMI = np.array(data[5])
            np.save(self._fileData, self._data)
            np.save(self._fileLabels, self._labels)
            np.save(self._fileDataHealthy, self._dataHealthy)
            np.save(self._fileLabelsHealthy, self._labelsHealthy)
            np.save(self._fileDataMI, self._dataMI)
            np.save(self._fileLabelsMI, self._labelsMI)
            print(self._name + ": Finished reimporting data. Saved to disk for next time.")
        self._data = np.multiply(self._data, 1.0 / 255.0)
        self._dataHealthy = np.multiply(self._dataHealthy, 1.0 / 255.0)
        self._dataMI = np.multiply(self._dataMI, 1.0 / 255.0)
