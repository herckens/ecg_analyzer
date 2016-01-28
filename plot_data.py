#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from dataset import DataSet

prefix = "../ptbdb/"

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
W = tf.Variable(tf.zeros([100, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
#y = tf.nn.softmax(tf.matmul(x, W))

# Define the cost function and training step.
#TODO cosmetics: follow naming convention
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

# Define the evaluation function.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.initialize_all_variables())
accuracyTrain = list()
accuracyTest = list()

# Train.
for i in range(500):
    batchXs, batchYs = trainData.next_batch(100)
    train_step.run(feed_dict = {x: batchXs, y_: batchYs})
    accuracyTrain.append(sess.run(accuracy, feed_dict={x: batchXs, y_: batchYs}))
    accuracyTest.append(sess.run(accuracy, feed_dict={x: testData.data, y_: testData.labels}))
valueW = sess.run(W)
print(valueW)
valueB = sess.run(b)
print(valueB)

batchXs, batchYs = trainData.next_batch(1000)
print("Accuracy (all) =        {}".format(sess.run(accuracy, feed_dict={x: batchXs, y_: batchYs})))
print("Specificity (healthy) = {}".format(sess.run(accuracy, feed_dict={x: testData.dataHealthy, y_: testData.labelsHealthy})))
print("Sensitivity (MI) =      {}".format(sess.run(accuracy, feed_dict={x: testData.dataMI, y_: testData.labelsMI})))

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
