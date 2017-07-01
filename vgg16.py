"""
VGG-16 implementation.
Based on: http://www.cs.toronto.edu/~frossard/post/vgg16/
"""

import tensorflow as tf
import numpy as np

import model

class VGG16(model.Model):
    """
    VGG16 class' definition.
    """

    #def __init__(self, x, y, learning_rate, trainable, is_training, threshold = 0.5, weights_file=None, sess=None):
    def __init__(self, x, y, learning_rate, trainable, threshold = 0.5, weights_file=None, sess=None):
        """
        VGG16's constructor.
        Inputs:
            x: Model's inputs (Float placeholder of images #[Batch size, height, width, channels])
            y: Groundtruth labels (Float placeholder of scalars #[Batch size])
            learning_rate: Learning rate for training (Float)
            trainable: Define if the VGG model should be fully retrained or not (Boolean)
            threshold: Threshold for output activations (Float)
            weights_file: Path to model's weights file (String)
            sess: A Tensorflow session (Tensorflow session)
        """

        super().__init__(x, y, learning_rate, threshold)
        #VGG construction
        self.trainable = trainable
        #self.is_training = is_training
        self.get_logits()
        self.get_probs()
        self.infer()
        #Pretrained weights load
        if weights_file is not None and sess is not None:
            weights = np.load(weights_file)
            keys = sorted(weights.keys())
            print('Loading pretrained VGG16 weights...')
            for i, k in enumerate(keys):
                if i < 30:
                    #print(i, k, np.shape(weights[k]))
                    sess.run(self.parameters[i].assign(weights[k]))
            print('Pretrained VGG16 weights loaded.')

    def get_logits(self):
        """
        Define the VGG16's computation graph.
        Return:
            logits: A tensor of floats corresponding to the computed logits (#[Batch size])
        """

        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            #normalized_inputs = tf.map_fn(lambda img: tf.image.per_image_standardization(img), self.inputs)
            normalized_inputs = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.inputs - mean

        with tf.name_scope('vgg') as scope:
            # conv1_1 [batch size, 224, 224, 3] -> [batch size, 224, 224, 64]
            with tf.name_scope('conv1_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1), trainable=self.trainable, name='weights')
                conv = tf.nn.conv2d(normalized_inputs, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=self.trainable, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv1_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv1_2 [batch size, 224, 224, 64] -> [batch size, 224, 224, 64]
            with tf.name_scope('conv1_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), trainable=self.trainable, name='weights')
                conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=self.trainable, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv1_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool1 [batch size, 224, 224, 64] -> [batch size, 112, 112, 64]
            self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

            # conv2_1 [batch size, 112, 112, 64] -> [batch size, 112, 112, 128]
            with tf.name_scope('conv2_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1), trainable=self.trainable, name='weights')
                conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=self.trainable, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv2_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv2_2 [batch size, 112, 112, 128] -> [batch size, 112, 112, 128]
            with tf.name_scope('conv2_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1), trainable=self.trainable, name='weights')
                conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=self.trainable, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv2_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool2 [batch size, 112, 112, 128] -> [batch size, 56, 56, 128]
            self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

            # conv3_1 [batch size, 56, 56, 128] -> [batch size, 56, 56, 256]
            with tf.name_scope('conv3_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1), trainable=self.trainable, name='weights')
                conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=self.trainable, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv3_2 [batch size, 56, 56, 256] -> [batch size, 56, 56, 256]
            with tf.name_scope('conv3_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), trainable=self.trainable, name='weights')
                conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=self.trainable, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv3_3 [batch size, 56, 56, 256] -> [batch size, 56, 56, 256]
            with tf.name_scope('conv3_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), trainable=self.trainable, name='weights')
                conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=self.trainable, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_3 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool3 [batch size, 56, 56, 256] -> [batch size, 28, 28, 256]
            self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

            # conv4_1 [batch size, 28, 28, 256] -> [batch size, 28, 28, 512]
            with tf.name_scope('conv4_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1), trainable=self.trainable, name='weights')
                conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=self.trainable, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv4_2 [batch size, 28, 28, 512] -> [batch size, 28, 28, 512]
            with tf.name_scope('conv4_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), trainable=self.trainable, name='weights')
                conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=self.trainable, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv4_3 [batch size, 28, 28, 512] -> [batch size, 28, 28, 512]
            with tf.name_scope('conv4_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), trainable=self.trainable, name='weights')
                conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=self.trainable, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_3 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool4 [batch size, 28, 28, 512] -> [batch size, 14, 14, 512]
            self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

            # conv5_1 [batch size, 14, 14, 512] -> [batch size, 14, 14, 512]
            with tf.name_scope('conv5_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), trainable=self.trainable, name='weights')
                conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=self.trainable, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv5_2 [batch size, 14, 14, 512] -> [batch size, 14, 14, 512]
            with tf.name_scope('conv5_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), trainable=self.trainable, name='weights')
                conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=self.trainable, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv5_3 [batch size, 14, 14, 512] -> [batch size, 14, 14, 512]
            with tf.name_scope('conv5_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), trainable=self.trainable, name='weights')
                conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=self.trainable, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_3 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool5 [batch size, 14, 14, 512] -> [batch size, 7, 7, 512]
            self.pool5 = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

            # fc1 [batch size, 7, 7, 512] -> [batch size, 1, 1, 4096]
            with tf.name_scope('fc1') as scope:
                shape = int(np.prod(self.pool5.get_shape()[1:]))
                fc1w = tf.Variable(tf.truncated_normal([shape, 4096], dtype=tf.float32, stddev=1e-1), trainable=self.trainable, name='weights')
                fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32), trainable=self.trainable, name='biases')
                pool5_flat = tf.reshape(self.pool5, [-1, shape])
                fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
                self.fc1 = tf.nn.relu(fc1l)
                self.parameters += [fc1w, fc1b]

            # fc2 [batch size, 1, 1, 4096] -> [batch size, 1, 1, 4096]
            with tf.name_scope('fc2') as scope:
                fc2w = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1),trainable=self.trainable, name='weights')
                fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32), trainable=self.trainable, name='biases')
                fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
                self.fc2 = tf.nn.relu(fc2l)
                self.parameters += [fc2w, fc2b]

            # fc3 [batch size, 1, 1, 4096] -> [batch size, 1, 1, 1]
            with tf.name_scope('fc3') as scope:
                fc3w = tf.Variable(tf.truncated_normal([4096, 1], dtype=tf.float32, stddev=1e-1), trainable=True, name='weights')
                fc3b = tf.Variable(tf.constant(1.0, shape=[1], dtype=tf.float32), trainable=True, name='biases')
                #batch_norm = tf.contrib.layers.batch_norm(self.fc2, is_training=self.is_training, updates_collections=None)
                self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
                self.parameters += [fc3w, fc3b]
                self.logits = tf.reshape(self.fc3l, [-1])

        return self.logits, normalized_inputs
