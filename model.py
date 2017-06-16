"""
Define a generic class that defines functions common to a large range of models
that could be used in this research.
"""

class Model:
    """
    Model class' definition.
    """

    def __init__(self, x, y, learning_rate, threshold = 0.5):
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
        self.labels = y
        self.trainable = trainable
        self.learning_rate = learning_rate
        self.process()
        self.predictions = tf.sigmoid(self.logits, name='predictions')
        self.threshold = tf.constant(threshold, dtype=tf.float32, name='detection_threshold')

    def process(self):
        """
        Define the model's computation graph.
        Return: The operation that computes logits.

        Should be redefined in a child class.
        """

        self.logits = 0

    def detect(self):
        """
        Classify model's inputs as normal/abnormal according to resulting logits.
        Return: A boolean tensor of predictions #[Batch size].
        """

        with tf.name_scope('detect'):
            detection = tf.greater_equal(self.predictions, self.threshold, name='detection')

        return detection

    def loss(self):
        """
        Compute the loss between the logits computed by the model and the groundtruth labels.
        Return: A scalar representing the loss #[1].
        """

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.labels, name='x_entropy')
            loss = tf.reduce_mean(cross_entropy, name='loss')

        return loss

    def accuracy(self):
        """
        Compute the accuracy of the model's logits according to the groundtruth labels.
        Return: A scalar representing the accuracy of the model #[1].
        """

        with tf.name_scope('accuracy'):
            prediction_to_float = tf.cast(self.detect(), dtype=tf.float32, name='detection_to_float')
            correct_prediction = tf.equal(prediction_to_float, self.y, name='count_correct_prediction')
            correct_prediction_to_float = tf.cast(correct_prediction, dtype=tf.float32, name='correct_prediction_to_float')
            accuracy = tf.reduce_mean(correct_prediction_to_float, name='accuracy')

        return accuracy

    def auc(self):
        """
        Compute the Area Under the ROC Curve (AUC) of the model.
        Return: A scalar representing the AUC of the model #[1].
        """

        with tf.name_scope('auc'):
            auc = tf.metrics.auc(y, self.predictions, num_thresholds=500, curve='ROC', name='auc')

        return auc

    def train(self):
        """
        Define the model's training operation.
        Return: The training operation.
        """

        with tf.name_scope('training'):
            learning_rate = self.learning_rate
            loss = self.loss()
            train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        return train

    def test(self):
        """
        Define the model's testing operations.
        Return: The cross entropy #[1], the number of correct predictions #[1], the AUC #[1]
        """

        with tf.name_scope('testing'):
            with tf.name_scope('cross_entropy'):
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.labels, name='x_entropy')
                sum_xent = tf.reduce_sum(cross_entropy, name='sum_xent')
            with tf.name_scope('accuracy'):
                prediction_to_float = tf.cast(self.detect(), dtype=tf.float32, name='detection_to_float')
                correct_prediction = tf.equal(prediction_to_float, self.y, name='count_correct_prediction')
                correct_prediction_to_float = tf.cast(correct_prediction, dtype=tf.float32, name='correct_prediction_to_float')
                true_count = tf.sum(correct_prediction_to_float, name='sum_correct_predictions')
            with tf.name_scope('auc'):
                auc = self.auc()

        return sum_xent, true_count, auc
