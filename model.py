"""
Define a generic class that defines functions common to a large range of models
that could be used in this research.
"""

import tensorflow as tf

class Model:
    """
    Model class' definition.
    """

    def __init__(self, x, y, learning_rate, threshold = 0.5):
        """
        Model's constructor.
        Inputs:
            x: Model's inputs (Float placeholder of images #[Batch size, height, width, channels])
            y: Groundtruth labels (Float placeholder of scalars #[Batch size])
            learning_rate: Learning rate for training (Float)
            threshold: Threshold for output activations (Float)
        """

        assert learning_rate > 0, 'The learning rate should be strictly positive'
        assert threshold >= 0, 'The threshold should be strictly positive or null'
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.inputs = x
        self.groundtruth = y

    def get_logits(self):
        """
        Define the model's computation graph.
        Return:
            logits: A tensor of floats corresponding to the computed logits

        Should be redefined in a child class.
        """

        self.logits = 0

        return self.logits

    def get_probs(self):
        """
        """

        with tf.name_scope('sigmoid'):
            self.activations = tf.sigmoid(tf.cast(self.logits, tf.float32), name='activations')
            self.reshaped_activations = tf.reshape(self.activations, shape=[-1], name='fix_shape_activations')

            return self.reshaped_activations

    def infer(self):
        """
        Classify the given inputs as normal/abnormal according to resulting logits.
        Return:
            inference: A boolean tensor of predictions (#[Batch size])
        """

        with tf.name_scope('infer'):
            self.inference = tf.greater_equal(self.reshaped_activations, self.threshold, name='inference')

        return self.inference

    def get_cross_entropy(self):
        """
        Compute the cross entropy between the logits computed by the model and the groundtruth labels.
        Return:
            cross_entropy: Scalar representing the cross_entropy (#[1])
        """

        with tf.name_scope('cross_entropy'):
            self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.groundtruth, name='x_entropy')

        return self.cross_entropy

    def get_loss_batch(self):
        """
        Compute the loss according to the current batch of images and groundtruth labels.
        Return:
            loss_batch: A float scalar representing the model's loss according to the current batch of images (#[1])
        """

        with tf.name_scope('loss_batch'):
            self.loss_batch = tf.reduce_mean(self.cross_entropy, name='loss_batch')

        return self.loss_batch

    def count_correct_prediction(self):
        """
        Compute the number of correct prediction of the model's logits according to the current batch of images and groundtruth labels.
        Return:
            correct_prediction_to_float: Scalar representing the number of model's correct prediction according to the current batch of images. (#[1])
        """

        with tf.name_scope('count_correct_prediction'):
            prediction_to_float = tf.cast(self.inference, dtype=tf.float32, name='inference_to_float')
            correct_prediction = tf.equal(prediction_to_float, self.groundtruth, name='count_correct_prediction')
            self.correct_prediction_to_float = tf.cast(correct_prediction, dtype=tf.float32, name='correct_prediction_to_float')

        return self.correct_prediction_to_float

    def get_accuracy_batch(self):
        """
        Compute the accuracy of the model according to the current batch of images and their groundtruth labels.
        Return:
            accuracy_batch: A float scalar representing the model's accuracy according to the current batch of images (#[1])
        """

        with tf.name_scope('accuracy_batch'):
            self.accuracy_batch = tf.reduce_mean(self.correct_prediction_to_float, name='accuracy_batch')

        return self.accuracy_batch

    def auc(self):
        """
        Compute the Area Under the ROC Curve (AUC) of the model.
        Return:
            auc: Scalar representing the AUC of the model (#[1])
        """

        with tf.name_scope('auc'):
            auc, update_op = tf.metrics.auc(self.groundtruth, self.inference, num_thresholds=200, curve='ROC', name='auc')

        return auc, update_op

    def train(self):
        """
        Define the model's training operation.
        Return:
            train: The training operation
        """

        with tf.name_scope('training'):
            #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            #with tf.control_dependencies(update_ops):
            #train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_batch)
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.learning_rate
            learning_rate = learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step, decay_steps=500, end_learning_rate=0.00001, power=0.5)
            # Passing global_step to minimize() will increment it at each step.
            #self.train = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss_batch, global_step=global_step)
            self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_batch, global_step=global_step)

        return self.train, learning_rate
