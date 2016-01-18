#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
import random
import tensorflow as tf

from database import DataBase
from dataconditioner import DataConditioner

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Convolution with stride of 1 and padding of 0.
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Pooling with max pooling over 2x2 blocks.
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Import the ECG data.
prefix = "../ptbdb/"
db = DataBase(prefix)
dc = DataConditioner()
data = list()
labels = list()
with open(prefix + "RECORDS") as f:
    for line in f:
        patientPath = line.rstrip('\n')
        ## Get diagnosis (pathologic or not).
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

# Separate into train and test data.
length = len(data)
#trainData = np.array(data[0 : int(length/2)])
#trainLabels = np.array(labels[0 : int(length/2)])
#testData = np.array(data[int(length/2) : int(length)])
#testLabels = np.array(labels[int(length/2) : int(length)])

#testData = np.array(data[0 : int(length/2)])
#testLabels = np.array(labels[0 : int(length/2)])
#trainData = np.array(data[int(length/2) : int(length)])
#trainLabels = np.array(labels[int(length/2) : int(length)])

# Select random samples for train and test datasets.
trainIndices = set()
testIndices = set()
allIndices = set(range(0, length))
trainData = list()
trainLabels = list()
testData = list()
testLabels = list()
while len(trainIndices) < int(length / 2):
    index = random.randint(0, length-1)
    if index not in trainIndices:
        trainIndices.add(index)
testIndices = allIndices.difference(trainIndices)
for i in trainIndices:
    trainData.append(data[i])
    trainLabels.append(labels[i])
for i in testIndices:
    testData.append(data[i])
    testLabels.append(labels[i])
trainData = np.array(trainData)
trainLabels = np.array(trainLabels)
testData = np.array(testData)
testLabels = np.array(testLabels)

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
for i in range(5000):
    train_step.run(feed_dict = {x: trainData, y_: trainLabels})

# Evaluate.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: testData, y_: testLabels}))

## Plot
#plt.close()
#plt.ion()
#for sample in trainData:
#    plt.plot(sample, 'b')
#for sample in testData:
#    plt.plot(sample, 'r')
#plt.show()
