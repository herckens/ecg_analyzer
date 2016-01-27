#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
import random
import tensorflow as tf

from database import DataBase
from dataconditioner import DataConditioner

prefix = "../ptbdb/"

fileTrainRecords = "RECORDS_TRAIN"
fileTestRecords = "RECORDS_TEST"

fileTrainData = "trainData.npy"
fileTrainLabels = "trainLabels.npy"
fileTrainDataHealthy = "trainDataHealthy.npy"
fileTrainLabelsHealthy = "trainLabelsHealthy.npy"
fileTrainDataMI = "trainDataMI.npy"
fileTrainLabelsMI = "trainLabelsMI.npy"
fileTestData = "testData.npy"
fileTestLabels = "testLabels.npy"
fileTestDataHealthy = "testDataHealthy.npy"
fileTestLabelsHealthy = "testLabelsHealthy.npy"
fileTestDataMI = "testDataMI.npy"
fileTestLabelsMI = "testLabelsMI.npy"

def shuffle_data(data, labels):
    length = len(data)
    indices = list(range(0, length))
    random.shuffle(indices)
    dataShuffled = list()
    labelsShuffled = list()
    for i in indices:
        dataShuffled.append(data[i])
        labelsShuffled.append(labels[i])
    return dataShuffled, labelsShuffled

def import_dataset(recordsFile):
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
    dataShuffled, labelsShuffled = shuffle_data(data, labels)
    dataHealthyShuffled, labelsHealthyShuffled = shuffle_data(dataHealthy, labelsHealthy)
    dataMIShuffled, labelsMIShuffled = shuffle_data(dataMI, labelsMI)
    return dataShuffled, labelsShuffled, dataHealthyShuffled, labelsHealthyShuffled, dataMIShuffled, labelsMIShuffled

try :
    # If previous data exists in files, load it.
    trainData = np.load(fileTrainData)
    trainLabels = np.load(fileTrainLabels)
    trainDataHealthy = np.load(fileTrainDataHealthy)
    trainLabelsHealthy = np.load(fileTrainLabelsHealthy)
    trainDataMI = np.load(fileTrainDataMI)
    trainLabelsMI = np.load(fileTrainLabelsMI)
    testData = np.load(fileTestData)
    testLabels = np.load(fileTestLabels)
    testDataHealthy = np.load(fileTestDataHealthy)
    testLabelsHealthy = np.load(fileTestLabelsHealthy)
    testDataMI = np.load(fileTestDataMI)
    testLabelsMI = np.load(fileTestLabelsMI)

    print("Loaded previous data")
except IOError:
    # If no previous files exist, reimport the ECG data.
    print("Reimporting data")
    # Import train data.
    train = import_dataset(fileTrainRecords)
    trainData = np.array(train[0])
    trainLabels = np.array(train[1])
    trainDataHealthy = np.array(train[2])
    trainLabelsHealthy = np.array(train[3])
    trainDataMI = np.array(train[4])
    trainLabelsMI = np.array(train[5])
    # Import test data.
    test = import_dataset(fileTestRecords)
    testData = np.array(test[0])
    testLabels = np.array(test[1])
    testDataHealthy = np.array(test[2])
    testLabelsHealthy = np.array(test[3])
    testDataMI = np.array(test[4])
    testLabelsMI = np.array(test[5])

    np.save(fileTrainData, trainData)
    np.save(fileTrainLabels, trainLabels)
    np.save(fileTrainDataHealthy, trainDataHealthy)
    np.save(fileTrainLabelsHealthy, trainLabelsHealthy)
    np.save(fileTrainDataMI, trainDataMI)
    np.save(fileTrainLabelsMI, trainLabelsMI)
    np.save(fileTestData, testData)
    np.save(fileTestLabels, testLabels)
    np.save(fileTestDataHealthy, testDataHealthy)
    np.save(fileTestLabelsHealthy, testLabelsHealthy)
    np.save(fileTestDataMI, testDataMI)
    np.save(fileTestLabelsMI, testLabelsMI)

# Start the session.
sess = tf.InteractiveSession()

# Placeholders for the data.
# x: input data
# y_: correct labels
x = tf.placeholder(tf.float32, [None, 100])
y_ = tf.placeholder(tf.float32, [None, 2])

# Define the model.
W = tf.Variable(tf.zeros([100, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define the cost function and training step.
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

# Define the evaluation function.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#TODO Don't throw away the train data.
trainData = trainDataHealthy.tolist()
trainData.extend(trainDataMI.tolist()[::5])
trainLabels = trainLabelsHealthy.tolist()
trainLabels.extend(trainLabelsMI.tolist()[::5])
trainData = np.array(trainData)
trainLabels = np.array(trainLabels)

sess.run(tf.initialize_all_variables())

trainData = np.multiply(trainData, 1.0 / 255.0)
testData = np.multiply(testData, 1.0 / 255.0)
testDataHealthy = np.multiply(testDataHealthy, 1.0 / 255.0)
testDataMI = np.multiply(testDataMI, 1.0 / 255.0)

accuracyTrain = list()
accuracyTest = list()

# Train.
for i in range(50):
    train_step.run(feed_dict = {x: trainData, y_: trainLabels})
    accuracyTrain.append(sess.run(accuracy, feed_dict={x: trainData, y_: trainLabels}))
    accuracyTest.append(sess.run(accuracy, feed_dict={x: testData, y_: testLabels}))
valueW = sess.run(W)
print(valueW)
valueB = sess.run(b)
print(valueB)
valueY = sess.run(y, feed_dict={x: testDataHealthy[0:5], y_: testLabelsHealthy[0:5]})
print("y_ = " + str(testLabelsHealthy[0:5]))
print("y = " + str(valueY))

print("Accuracy (all) =        {}".format(sess.run(accuracy, feed_dict={x: testData, y_: testLabels})))
print("Specificity (healthy) = {}".format(sess.run(accuracy, feed_dict={x: testDataHealthy, y_: testLabelsHealthy})))
print("Sensitivity (MI) =      {}".format(sess.run(accuracy, feed_dict={x: testDataMI, y_: testLabelsMI})))

sess.close()

## Plot
plt.close()
plt.ion()
plt.plot(accuracyTrain, 'b')
plt.plot(accuracyTest, 'r')
#for sample in trainData[0:3]:
#    plt.plot(sample)
#for sample in testData:
#    plt.plot(sample, 'r')
plt.show()
