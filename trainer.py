"""
"""

import tensorflow as tf
import argparse

import vgg16

def _parse_function(example_proto):
    """
    """

    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""), "label": tf.FixedLenFeature((), tf.int32, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)

    return parsed_features["image"], parsed_features["label"]

def main(args):
    """
    """

    #Create dataset
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(args.batch_size)
    dataset = dataset.map(_parse_function)
    #Create iterator
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    #Instantiate session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    #Instantiate model
    model = vgg16.VGG16(next_element[0], args.learning_rate, args.trainable, threshold=args.threshold, weights_file=args.vgg_weights, sess=sess)
    loss = model.loss(next_element[0], next_element[1])
    accuracy = model.accuracy(next_element[0], next_element[1])
    auc = model.auc(next_element[0], next_element[1])
    train = model.train(next_element[0], next_element[1])
    #Create summaries
    #TODO

    for epoch in range(args.epochs):
        #Training
        step = 0
        while True:
            training_filenames = [args.train_records]
            sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
            try:
                sess.run([train, loss, accuracy, auc])
                if step % args.summary_step is 0:
                    #TODO: SUMMARY
                    pass
                step += 1
            except tf.errors.OutOfRangeError:
                print('Epoch %d complete'%(epoch))
                break
        #Validation
        while True:
            validation_filenames = [args.val_records]
            sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
            try:
                sess.run([loss, accuracy, auc])
                #TODO: SUMMARY
            except tf.errors.OutOfRangeError:
                break
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments for the model\'s trainer')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--ftrain', dest='trainable', type=bool, default=False, help='Full train (VGG)')
    parser.add_argument('--weights', dest='vgg_weights', type=str, default=False, help='Path to the VGG\'s pretrained weights')
    parser.add_argument('--thr', dest='threshold', type=float, default=0.5, help='Model\'s detection threshold')
    parser.add_argument('--trecord', dest='train_records', type=str, default='', help='Path to trainset tfrecords')
    parser.add_argument('--vrecord', dest='val_records', type=str, default='', help='Path to valset tfrecords')
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--sumstep', dest='summary_step', type=int, default=50, help='Number of summary steps')
    parser.add_argument('--bs', dest='batch_size', type=int, default=20, help='Mini batch size')
    args = parser.parse_args()
    main(args)
