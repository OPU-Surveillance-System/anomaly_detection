"""
Define a generic class that defines functions common to a large range of models
that could be used in this research.
"""

import tensorflow as tf

class Model:
    """
    Model class' definition.
    """

    def __init__(self, x, learning_rate, threshold = 0.5):
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
        self.inputs = x
        self.learning_rate = learning_rate
        self.threshold = tf.constant(threshold, dtype=tf.float32, name='detection_threshold')

    def process(self):
        """
        Define the model's computation graph.
        Return:
            logits: The operation that computes logits.

        Should be redefined in a child class.
        """

        logits = 0

        return logits

    def infer(self):
        """
        Classify the given inputs as normal/abnormal according to resulting logits.
        Return:
            inference: A boolean tensor of predictions #[Batch size].
        """

        with tf.name_scope('infer'):
            logits = self.process()
            activations = tf.sigmoid(tf.cast(logits, tf.float32), name='activations')
            reshaped_activations = tf.reshape(activations, shape=[-1], name='fix_shape_activations')
            inference = tf.greater_equal(activations, self.threshold, name='inference')

        return inference

    def loss(self, y):
        """
        Compute the loss between the logits computed by the model and the groundtruth labels.
        Inputs:
            y: Groundtruth labels (Float placeholder of labels #[Batch size])
        Return:
            loss: Scalar representing the loss #[1].
        """

        with tf.name_scope('loss'):
            logits = self.process()
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y, name='x_entropy')
            loss = tf.reduce_mean(cross_entropy, name='loss')

        return loss

    def accuracy(self, y):
        """
        Compute the accuracy of the model's logits according to the groundtruth labels.
        Inputs:
            y: Groundtruth labels (Float placeholder of labels #[Batch size])
        Return:
            accuracy: Scalar representing the accuracy of the model #[1].
        """

        with tf.name_scope('accuracy'):
            inference = self.infer()
            prediction_to_float = tf.cast(inference, dtype=tf.float32, name='inference_to_float')
            correct_prediction = tf.equal(prediction_to_float, y, name='count_correct_prediction')
            correct_prediction_to_float = tf.cast(correct_prediction, dtype=tf.float32, name='correct_prediction_to_float')
            accuracy = tf.reduce_mean(correct_prediction_to_float, name='accuracy')

        return accuracy

    def auc(self, y):
        """
        Compute the Area Under the ROC Curve (AUC) of the model.
        Inputs:
            y: Groundtruth labels (Float placeholder of labels #[Batch size])
        Return:
            auc: Scalar representing the AUC of the model #[1].
        """

        with tf.name_scope('auc'):
            inference = self.infer()
            auc = tf.metrics.auc(y, inference, num_thresholds=500, curve='ROC', name='auc')

        return auc

    def train(self, y):
        """
        Define the model's training operation.
        Inputs:
            y: Groundtruth labels (Float placeholder of labels #[Batch size])
        Return:
            train: The training operation.
        """

        with tf.name_scope('training'):
            learning_rate = self.learning_rate
            loss = self.loss(x, y)
            train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        return train
