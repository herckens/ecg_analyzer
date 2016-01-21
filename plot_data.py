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
fileTestData = "testData.npy"
fileTestLabels = "testLabels.npy"

def import_dataset(recordsFile):
    db = DataBase(prefix)
    dc = DataConditioner()
    data = list()
    labels = list()
    with open(prefix + recordsFile) as f:
        for line in f:
            patientPath = line.rstrip('\n')
            print(patientPath)
            diagnosis = db.get_diagnosis(patientPath)
            if diagnosis == 'Healthy control':
                lead1_raw = db.get_data(patientPath, lead = 0)
                slices = dc.condition_signal(lead1_raw)
                data += slices
                for i in range(0, len(slices)):
                    labels.append([1,0])
            elif diagnosis == 'Myocardial infarction':
                lead1_raw = db.get_data(patientPath, lead = 0)
                slices = dc.condition_signal(lead1_raw)
                data += slices
                for i in range(0, len(slices)):
                    labels.append([0,1])
            else :
                continue
    # Shuffle the data.
    length = len(data)
    indices = list(range(0, length))
    random.shuffle(indices)
    dataShuffled = list()
    labelsShuffled = list()
    for i in indices:
        dataShuffled.append(data[i])
        labelsShuffled.append(labels[i])
    return dataShuffled, labelsShuffled

try :
    # If previous data exists in files, load it.
    trainData = np.load(fileTrainData)
    trainLabels = np.load(fileTrainLabels)
    testData = np.load(fileTestData)
    testLabels = np.load(fileTestLabels)
    print("Loaded previous data")
except IOError:
    # If no previous files exist, reimport the ECG data.
    print("Reimporting data")
    trainData, trainLabels = import_dataset(fileTrainRecords)
    testData, testLabels = import_dataset(fileTestRecords)

    np.save(fileTrainData, trainData)
    np.save(fileTrainLabels, trainLabels)
    np.save(fileTestData, testData)
    np.save(fileTestLabels, testLabels)

# Start the session.
sess = tf.InteractiveSession()

# Placeholders for the data.
# x: input images
# y_: correct labels
x = tf.placeholder(tf.float32, [None, 100])
y_ = tf.placeholder(tf.float32, [None, 2])

# Define the model.
W = tf.Variable(tf.zeros([100, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define the cost function and training step.
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess.run(tf.initialize_all_variables())

# Train.
#for i in range(2):
#    train_step.run(feed_dict = {x: trainData, y_: trainLabels})

#half = int(len(trainData)/2)
#train_step.run(feed_dict = {x: trainData[0:half], y_: trainLabels[0:half]})
#train_step.run(feed_dict = {x: trainData[half+1:], y_: trainLabels[half+1:]})
#train_step.run(feed_dict = {x: trainData[0:i], y_: trainLabels[0:i]})
train_step.run(feed_dict = {x: trainData, y_: trainLabels})
#train_step.run(feed_dict = {x: trainData[1:2], y_: trainLabels[1:2]})
valueW = sess.run(W)
print(valueW)
valueB = sess.run(b)
print(valueB)

# Evaluate.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#print("i = {}, accuracy = {}".format(i, sess.run(accuracy, feed_dict={x: testData, y_: testLabels})))
print("accuracy = {}".format(sess.run(accuracy, feed_dict={x: testData, y_: testLabels})))

sess.close()

## Plot
plt.close()
plt.ion()
for sample in trainData[0:3]:
    plt.plot(sample)
#for sample in testData:
#    plt.plot(sample, 'r')
plt.show()
