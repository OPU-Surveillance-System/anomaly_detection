"""
Define a generic class that defines functions common to a large range of models
that could be used in this research.
"""

import tensorflow as tf

class Model:
    """
    Model class' definition.
    """

    def __init__(self, learning_rate, threshold = 0.5):
        """
        Model's constructor.
        Inputs:
            x: Model's inputs (Float placeholder of images #[Batch size, height, width, channels])
            y: Groundtruth labels (Float placeholder of labels #[Batch size])
            learning_rate: Learning rate for training (Float)
            threshold: Threshold for output activations (Float)
        """

        assert learning_rate > 0, 'The learning rate should be strictly positive'
        assert threshold >= 0, 'The threshold should be strictly positive or null'
        self.learning_rate = learning_rate
        self.threshold = tf.constant(threshold, dtype=tf.float32, name='detection_threshold')

    def process(self, x):
        """
        Define the model's computation graph.
        Return:
            logits: The operation that computes logits.

        Should be redefined in a child class.
        """

        logits = 0

        return logits

    def infer(self, x):
        """
        Classify the given inputs as normal/abnormal according to resulting logits.
        Inputs:
            x: Model's inputs (Float placeholder of images #[Batch size, height, width, channels])
        Return:
            inference: A boolean tensor of predictions #[Batch size]
        """

        with tf.name_scope('infer'):
            activations = tf.sigmoid(tf.cast(self.logits, tf.float32), name='activations')
            reshaped_activations = tf.reshape(activations, shape=[-1], name='fix_shape_activations')
            inference = tf.greater_equal(activations, self.threshold, name='inference')

        return inference

    def cross_entropy(self, y):
        """
        Compute the cross entropy between the logits computed by the model and the groundtruth labels.
        Inputs:
            y: Groundtruth labels (Float placeholder of labels #[Batch size])
        Return:
            cross_entropy: Scalar representing the cross_entropy #[1]
        """

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=y, name='x_entropy')

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

    def auc(self, x, y):
        """
        Compute the Area Under the ROC Curve (AUC) of the model.
        Inputs:
            x: Model's inputs (Float placeholder of images #[Batch size, height, width, channels])
            y: Groundtruth labels (Float placeholder of labels #[Batch size])
        Return:
            auc: Scalar representing the AUC of the model #[1]
        """

        with tf.name_scope('auc'):
            inference = self.infer(x)
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
