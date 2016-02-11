#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import time

from dataset import DataSet

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))

prefix = "../ptbdb/"
numHidden = 300

# Load the train and test datasets.
trainData = DataSet("train", prefix)
testData = DataSet("test", prefix)

# Start the session.
sess = tf.InteractiveSession()

# Placeholders for the data.
# x: input data
# y_: correct labels
x = tf.placeholder(tf.float32, [None, 100])
y_ = tf.placeholder(tf.float32, [None, 2])

# Define the model.
wHidden = init_weights([100,numHidden])
bHidden = tf.Variable(tf.zeros([numHidden]))
hidden = tf.nn.tanh(tf.matmul(x,wHidden) + bHidden)
wOut = tf.Variable(tf.zeros([numHidden, 2]))
bOut = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(hidden, wOut) + bOut)
#y = tf.nn.softmax(tf.matmul(x, w))

# Define the cost function and training step.
crossEntropy = -tf.reduce_sum(y_*tf.log(y))
learningRate = 0.0001
trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(crossEntropy)

# Define the evaluation function.
correctPrediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction, "float"))

sess.run(tf.initialize_all_variables())
accuracyTrain = list()
accuracyTest = list()

# Train.
start = time.time()
print("{}: Start training.".format(start))
for i in range(10000):
    batchXs, batchYs = trainData.next_batch(256)
    trainStep.run(feed_dict = {x: batchXs, y_: batchYs})
    if i%100 == 0:
        batchXs, batchYs = trainData.next_batch(4096)
        accuracyTrain.append(sess.run(accuracy, feed_dict={x: batchXs, y_: batchYs}))
        batchXs, batchYs = testData.next_batch(4096)
        accuracyTest.append(sess.run(accuracy, feed_dict={x: batchXs, y_: batchYs}))
        # Decay learning rate over time.
        learningRate *= 0.99
        trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(crossEntropy)
end = time.time()
print("{}: Stop training.".format(end))
valueW = sess.run(wOut)
print(valueW)
valueB = sess.run(bOut)
print(valueB)
print("Time for training = {}".format(end - start))

batchXs, batchYs = testData.next_batch(5000)
print("Accuracy (all) =        {}".format(sess.run(accuracy, feed_dict={x: batchXs, y_: batchYs})))
print("Specificity (healthy) = {}".format(sess.run(accuracy, feed_dict={x: testData.dataHealthy, y_: testData.labelsHealthy})))
print("Sensitivity (MI) =      {}".format(sess.run(accuracy, feed_dict={x: testData.dataMI, y_: testData.labelsMI})))

sess.close()

w0 = list()
w1 = list()
for elem in valueW:
    w0.append(elem[0])
    w1.append(elem[1])

# Plot
plt.close()
plt.ion()
plt.figure(1)
plt.subplot(211)
plt.plot(accuracyTrain, 'b', label = 'Train')
plt.plot(accuracyTest, 'r', label = 'Test')
plt.legend()
plt.subplot(212)
plt.plot(w0, 'b')
plt.plot(w1, 'r')
#for sample in trainData[0:3]:
#    plt.plot(sample)
#for sample in testData:
#    plt.plot(sample, 'r')
plt.show()
