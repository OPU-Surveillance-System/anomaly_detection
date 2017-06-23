"""
VGG-16 implementation.
Based on: http://www.cs.toronto.edu/~frossard/post/vgg16/
"""

import tensorflow as tf
import numpy as np

import model

class VGG16():
    """
    VGG16 class' definition.
    """

    def __init__(self, x, learning_rate, trainable, threshold = 0.5, weights_file=None, sess=None):
        """
        VGG16's constructor.
        Inputs:
            x: Model's inputs (Float placeholder of images #[Batch size, height, width, channels])
            learning_rate: Learning rate for training (Float)
            trainable: Define if the VGG model should be fully retrained or not
            threshold: Threshold for output activations (Float)
            weights_file: Path to model's weights file
            sess: A Tensorflow session
        """

        assert learning_rate > 0, 'The learning rate should be strictly positive'
        assert threshold >= 0, 'The threshold should be strictly positive or null'
        self.learning_rate = learning_rate
        self.threshold = tf.constant(threshold, dtype=tf.float32, name='detection_threshold')
        #VGG construction
        self.trainable = trainable
        self.logits = self.process(x)
        if weights_file is not None and sess is not None:
            weights = np.load(weights_file)
            keys = sorted(weights.keys())
            for i, k in enumerate(keys):
                if i < 30:
                    print(i, k, np.shape(weights[k]))
                    sess.run(self.parameters[i].assign(weights[k]))

    def process(self, x):
        """
        Define the VGG16's computation graph.
        Inputs:
            x: Model's inputs (Float placeholder of images #[Batch size, height, width, channels])
        Return:
            The operation that computes logits.
        """

        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            images = tf.map_fn(lambda img: tf.image.per_image_standardization(img), x)

        with tf.name_scope('vgg') as scope:
            # conv1_1 [batch size, 224, 224, 3] -> [batch size, 224, 224, 64]
            with tf.name_scope('conv1_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1), trainable=self.trainable, name='weights')
                conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
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
                self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
                self.parameters += [fc3w, fc3b]
                logits = tf.reshape(self.fc3l, [-1])

        return logits

    def infer(self, logits, x):
        """
        Classify the given inputs as normal/abnormal according to resulting logits.
        Inputs:
            x: Model's inputs (Float placeholder of images #[Batch size, height, width, channels])
        Return:
            inference: A boolean tensor of predictions #[Batch size]
        """

        with tf.name_scope('infer'):
            activations = tf.sigmoid(tf.cast(logits, tf.float32), name='activations')
            reshaped_activations = tf.reshape(activations, shape=[-1], name='fix_shape_activations')
            inference = tf.greater_equal(activations, self.threshold, name='inference')

        return inference

    def cross_entropy(self, logits, y):
        """
        Compute the cross entropy between the logits computed by the model and the groundtruth labels.
        Inputs:
            y: Groundtruth labels (Float placeholder of labels #[Batch size])
        Return:
            cross_entropy: Scalar representing the cross_entropy #[1]
        """

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y, name='x_entropy')

        return cross_entropy

    def count_correct_prediction(self, prediction, y):
        """
        Compute the number of correct prediction of the model's logits according to the groundtruth labels.
        Inputs:
            prediction: Model's prediction (Float placeholder of images #[Batch size])
            y: Groundtruth labels (Float placeholder of labels #[Batch size])
        Return:
            accuracy: Scalar representing the accuracy of the model #[1]
        """

        with tf.name_scope('accuracy'):
            prediction_to_float = tf.cast(prediction, dtype=tf.float32, name='inference_to_float')
            correct_prediction = tf.equal(prediction_to_float, y, name='count_correct_prediction')
            correct_prediction_to_float = tf.cast(correct_prediction, dtype=tf.float32, name='correct_prediction_to_float')

        return correct_prediction_to_float

    def auc(self, inference, y):
        """
        Compute the Area Under the ROC Curve (AUC) of the model.
        Inputs:
            x: Model's inputs (Float placeholder of images #[Batch size, height, width, channels])
            y: Groundtruth labels (Float placeholder of labels #[Batch size])
        Return:
            auc: Scalar representing the AUC of the model #[1]
        """

        with tf.name_scope('auc'):
            auc, update_op = tf.metrics.auc(y, inference, num_thresholds=200, curve='ROC', name='auc')

        return auc, update_op

    def train(self, cross_entropy):
        """
        Define the model's training operation.
        Inputs:
            cross_entropy: cross entropy operation over a batch #[1]
        Return:
            train: The training operation
        """

        with tf.name_scope('training'):
            loss = tf.reduce_mean(cross_entropy, name='loss')
            train = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        return train
