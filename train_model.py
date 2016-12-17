# Load pickled data
import pickle
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
training_file = './traffic-signs-data/train.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

def preprocess_features(X_train):

    new_images = np.zeros(X_train.shape[0:-1])
    for i in range(X_train.shape[0]):
        new_images[i] = cv2.cvtColor(X_train[i], cv2.COLOR_RGB2GRAY)
    new_images = new_images/255

    return new_images[:,:,:,None]

def preprocess_target(y_train):

    new_y_train = np.zeros((y_train.shape[0], 43))

    for i in range(y_train.shape[0]):
        new_y_train[i][y_train[i]] = 1.0

    return new_y_train

X_train_gray, X_test_gray = preprocess_features(X_train), preprocess_features(X_test)

y_test = preprocess_target(y_test)

### Generate data additional (if you want to!)
def distort_data(X_train):
    #Translation
    examples,row,col,__ = X_train.shape
    t_sample = np.random.uniform(low=-0.1, high=0.1)
    r_sample = np.random.uniform(low=-10, high=10 )
    M_t = np.float32([[1,0,math.floor(col*(1+t_sample))],[0,1,math.floor(row*(1+t_sample))]])
    M_r = cv2.getRotationMatrix2D((col//2,row//2),r_sample,1)
    for i in range(examples):
        img = cv2.warpAffine(X_train[i,:,:,0],M_t,(col,row))
        img = cv2.warpAffine(img,M_r,(col,row))
        X_train[i,:,:,0] = img
    return X_train

### and split the data into training/validation/testing sets here.
X_train_gray , X_dev_gray, y_train, y_dev = train_test_split(X_train_gray, y_train, test_size = 0.10, stratify = y_train)
y_train, y_dev = preprocess_target(y_train), preprocess_target(y_dev)

### Define your architecture here.
_, image_height, image_width, color_channels = X_train_gray.shape
x = tf.placeholder(tf.float32, shape=(None, image_width, image_height, color_channels))
y = tf.placeholder(tf.int32, shape=(None,43))
keep_prob = tf.placeholder(tf.float32)

def SyedNet(x, keep_prob):

    x = tf.image.resize_images(x, (32, 32))
    layer_width = {
    'layer 1': 100,
    'layer 2': 150,
    'layer 3': 200,
    'layer 4': 43
    }
    weights = {
    'layer 1': tf.Variable(tf.truncated_normal(stddev=0.01,shape=(3,3,1,layer_width['layer 1']))),
    'layer 2': tf.Variable(tf.truncated_normal(stddev=0.01,shape=(5,5,layer_width['layer 1'], layer_width['layer 2']))),
    'layer 3': tf.Variable(tf.truncated_normal(stddev=0.01,shape=(15*15*layer_width['layer 1']+ 5*5*layer_width['layer 2'], layer_width['layer 3']))),
    'layer 4': tf.Variable(tf.truncated_normal(stddev=0.01,shape=(layer_width['layer 3'], layer_width['layer 4'])))
    }
    bias = {
    'layer 1': tf.Variable(tf.zeros(layer_width['layer 1'])),
    'layer 2': tf.Variable(tf.zeros(layer_width['layer 2'])),
    'layer 3': tf.Variable(tf.zeros(layer_width['layer 3'])),
    'layer 4': tf.Variable(tf.zeros(layer_width['layer 4'])),
    }
    #layer 1
    conv1 = tf.nn.conv2d(x, weights['layer 1'], strides=[1,1,1,1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, bias['layer 1'])
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides=[1, 2, 2, 1], padding='VALID')
    #convi = tf.nn.dropout(conv1, keep_prob)
    #layer 2
    conv2 = tf.nn.conv2d(conv1, weights['layer 2'], strides=[1,1,1,1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, bias['layer 2'])
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides=[1, 2, 2, 1], padding='VALID')
    #conv2 = tf.nn.dropout(conv2, keep_prob)
    #layer 3
    fl_con = tf.concat(1, [flatten(conv2),flatten(conv1)])
    fl_con = tf.matmul(fl_con, weights['layer 3']) + bias['layer 3']
    fl_con = tf.nn.relu(fl_con)
    #conv2 = tf.nn.dropout(conv2, keep_prob)
    #layer 4
    return tf.matmul(fl_con, weights['layer 4']) + bias['layer 4']

out = SyedNet(x, keep_prob)
softmax = tf.nn.softmax(out)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, y))
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op)
correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

EPOCHS = 120
BATCH_SIZE = 128
DEV_BATCH_SIZE = 128
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    steps_per_epoch = X_train.shape[0] // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    index = np.arange(X_train_gray.shape[0])
    dev_index = np.arange(X_dev_gray.shape[0])
    for i in range(EPOCHS):
        np.random.shuffle(index)
        np.random.shuffle(dev_index)
        if i==100:
            X_train_gray = distort_data(X_train_gray)
        for step in range(steps_per_epoch):
            start, end = step*BATCH_SIZE, (step+1)*BATCH_SIZE
            batch_x, batch_y = X_train_gray[index[start:end]], y_train[index[start:end]]
            train_loss, _ = sess.run([loss_op, train_op], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            print("Train loss = {}".format(train_loss))
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x: X_dev_gray[dev_index[0:DEV_BATCH_SIZE+1]], y: y_dev[0:DEV_BATCH_SIZE+1], keep_prob: 1.0})

        print("EPOCH {} ...".format(i+1))
        print("Train loss = {}".format(train_loss))
        print("Validation loss = {}".format(loss))
        print("Validation accuracy = {}".format(acc))

        #test_loss, test_acc = sess.run([loss_op, accuracy_op], feed_dict={x: X_test_gray, y: y_test, keep_prob: 1.0})

        #print("Test loss = {}".format(test_loss))
        #print("Test accuracy = {}".format(test_acc))

    saver.save(session, 'saved_vars')
