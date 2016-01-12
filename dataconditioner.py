import numpy as np
from scipy import signal

class DataConditioner:
    def __init__(self):
        pass

    def remove_drift(self, inputData):
        """
        Remove baseline drift from inputData.
        """
        Wn = 0.001
        b, a = signal.butter(3, Wn, 'lowpass', analog=False)
        baseline = signal.filtfilt(b, a, inputData)
        outputData = inputData - baseline
        return outputData

    def find_peaks(self, data):
        """
        Returns a np.array with the indices of all local maxima in data.
        """
        peakInd = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1 # local max_
        return peakInd

    def find_peak_threshold(self, peakInd, data, noOfPeaks):
        """
        Find the noOfPeaks highest peaks within data.
        Return 70% of the average of these peaks.
        """
        highestPeaks = list()
        #highestPeaks = [(0,0)]
        for index in peakInd:
            if len(highestPeaks) < 5:
                highestPeaks.append(data[index])
                next
            for peak in highestPeaks:
                #print('data = ' + str(data[index]))
                #print('peak[1] = ' + str(peak[1]))
                if data[index] > peak:
                    #print('bigger found')
                    highestPeaks[highestPeaks.index(peak)] = data[index]
                    break
        arr = np.array(highestPeaks)
        return 0.7 * arr.mean()

    def remove_small_peaks(self, peakInd, data, threshold):
        """
        Returns a list of all items from peakInd that match the condition
        data[item] > threshold.
        """
        highPeaks = list(filter(lambda item: data[item] > threshold, peakInd))
        return highPeaks

    def find_r_peaks(self, data):
        """
        Returns a list of indices that mark the temporal location of R peaks in
        data. Data is assumed to be a standard ECG lead 1 signal.
        """
        # Find all peaks.
        peakInd = self.find_peaks(data)
        # Find threshold for R peaks.
        threshold = self.find_peak_threshold(peakInd, data, 5)
        # Accept peaks if higher than threshold.
        rPeaksInd = self.remove_small_peaks(peakInd, data, threshold)
        rPeaksVal = data[rPeaksInd]
        return rPeaksInd, rPeaksVal

    def calc_avg_time_between_beats(self, rPeaksInd):
        """
        Calculate the average time between two consecutive peaks in rPeaksInd..
        """
        periods = list()
        for i in range(0,len(rPeaksInd)-1):
            periods.append(rPeaksInd[i+1] - rPeaksInd[i])
        periods = np.array(periods)
        return periods.mean()
