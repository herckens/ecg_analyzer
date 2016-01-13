import numpy as np
from scipy import signal, ndimage

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
        for index in peakInd:
            if len(highestPeaks) < 5:
                highestPeaks.append(data[index])
                next
            for peak in highestPeaks:
                if data[index] > peak:
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

    def cut_data_into_beats(self, data, rPeaksInd):
        """
        Cut the signal into single heart beats, centered around the R peak.
        Return a list of signal slices.
        The first and last R peak in the signal are discarded.
        """
        beatPeriod = self.calc_avg_time_between_beats(rPeaksInd)
        slices = list()
        for peak in rPeaksInd[1:-1]:
            start = peak - int(beatPeriod / 2)
            end = peak + int(beatPeriod / 2)
            slices.append(data[start:end])
        return slices

    def zoom_to_length(self, data, length):
        """
        Zoom all 1D arrays in the list data so that their length equals length.
        """
        dataOut = list()
        for arr in data:
            zoomFactor = length / len(arr)
            new = ndimage.zoom(arr, zoomFactor, order=1)
            dataOut.append(new)
        return dataOut
    
    def scale_to_int(self, data, resolution):
        dataOut = list()
        for arr in data:
            scaled = arr * resolution
            rounded = np.rint(scaled)
            dataOut.append(rounded.astype(int))
        return dataOut

    def condition_signal(self, data):
        """
        Condition a continuous ECG lead I signal so that it can be used as
        samples to train a classifier.
        Returns a list of arrays where each array is a signal slice of one heart beat,
        centered around the R peak.
        """
        # Remove baseline drift.
        lead1_driftless = self.remove_drift(data)
        # Smooth the data.
        Wn = 0.08
        b, a = signal.butter(6, Wn, 'lowpass', analog=False)
        lead1_smoothed = signal.filtfilt(b, a, lead1_driftless)
        # Find location and value of R peaks.
        rPeaksInd, rPeaksVal = self.find_r_peaks(lead1_smoothed)
        # Cut the signal into individual heart beats centered around R peaks.
        slices = self.cut_data_into_beats(lead1_smoothed, rPeaksInd)
        slicesZoomed = self.zoom_to_length(slices, 100)
        slicesScaled = self.scale_to_int(slicesZoomed, 100)
        return slicesScaled
