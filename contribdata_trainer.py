"""
Train a given model according to a given dataset.
"""

import os
import tensorflow as tf
from tensorflow.python.framework import ops
import argparse
import matplotlib.pyplot as plt

import vgg16

def _parse_function(example_proto):
    """
    Parse a given tfrecord's entry into an image and a label.
    Inputs:
        example_proto: A tfrecord's entry.
    Returns:
        image_tofloat: A tensor representing the image.
        preproc_label: A tensor representing the label.
    """

    features = {'height': tf.FixedLenFeature((), tf.int64, default_value=0),
                'width': tf.FixedLenFeature((), tf.int64, default_value=0),
                'label': tf.FixedLenFeature((), tf.int64, default_value=0),
                'image': tf.FixedLenFeature((), tf.string, default_value="")}
    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.decode_raw(parsed_features['image'], tf.uint8)
    image.set_shape([224 * 224 * 3])
    image_resized = tf.reshape(image, shape=[224, 224, 3])
    image_tofloat = tf.cast(image_resized, tf.float32)
    preproc_label = tf.reshape(tf.cast(parsed_features["label"], tf.float32), shape=[-1])

    return image_tofloat, preproc_label

def main(args):
    #Create output directories
    if not os.path.exists(args.exp_out):
        os.makedirs(args.exp_out)
    if not os.path.exists(args.exp_out + '/logs'):
        os.makedirs(args.exp_out + '/logs')
    if not os.path.exists(args.exp_out + '/serial'):
        os.makedirs(args.exp_out + '/serial')
    #Create dataset
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    #dataset = dataset.repeat(args.epoch)
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(args.batch_size)
    #Create iterator
    iterator = dataset.make_initializable_iterator()
    image, label = iterator.get_next()
    #Instantiate session
    sess = tf.InteractiveSession()
    #Instantiate model and define operations
    model = vgg16.VGG16(image, args.learning_rate, args.trainable, threshold=args.threshold, weights_file=args.vgg_weights, sess=sess)
    cross_entropy = model.cross_entropy(label)
    prediction = model.infer(image)
    correct_prediction = model.count_correct_prediction(prediction, label)
    train = model.train(cross_entropy)
    #Create summaries
    pl_loss = tf.placeholder(tf.float32, name='loss_placeholder')
    pl_accuracy = tf.placeholder(tf.float32, name='accuracy_placeholder')
    with tf.variable_scope("train_set"):
        t_loss_summary = tf.summary.scalar(tensor=pl_loss, name='loss')
        t_accuracy_summary = tf.summary.scalar(tensor=pl_accuracy, name='accuracy')
        t_summaries = tf.summary.merge([t_loss_summary, t_accuracy_summary])
    with tf.variable_scope("validation_set"):
        v_loss_summary = tf.summary.scalar(tensor=pl_loss, name='loss')
        v_accuracy_summary = tf.summary.scalar(tensor=pl_accuracy, name='accuracy')
        v_summaries = tf.summary.merge([v_loss_summary, v_accuracy_summary])
    train_writer = tf.summary.FileWriter(os.path.join(args.exp_out, 'logs/train'), sess.graph)
    validation_writer = tf.summary.FileWriter(os.path.join(args.exp_out, 'logs/validation'), sess.graph)
    #Create saver
    saver = tf.train.Saver()
    #Init variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    #Training loop
    batch_count = 0
    for epoch in range(args.epochs):
        #Training
        training_filenames = [args.train_records]
        sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
        step = 0
        while True:
            try:
                _, t_loss, t_accuracy, count_1 = sess.run([train, tf.reduce_mean(cross_entropy), tf.reduce_mean(correct_prediction), tf.reduce_sum(tf.cast(prediction, tf.float32))])
                if step % args.summary_step == 0:
                    print('epoch %d, step %d (%d images), loss: %.4f, accuracy: %.4f (%d alarms)'%(epoch, step, (step + 1) * 20, t_loss, t_accuracy, count_1))
                    feed_dict = {pl_loss: t_loss, pl_accuracy: t_accuracy}
                    train_str = sess.run(t_summaries, feed_dict=feed_dict)
                    train_writer.add_summary(train_str, batch_count)
                    train_writer.flush()
                step += 1
                batch_count += 1
            except tf.errors.OutOfRangeError:
                print('Epoch %d complete'%(epoch))
                break
        #Save model
        if epoch % args.save_epoch is 0:
            save_path = saver.save(sess, os.path.join(args.exp_out, 'serial/model.ckpt'), global_step=epoch)
            print('Model saved in file: %s'%(save_path))
        #Validation
        # validation_filenames = [args.val_records]
        # sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
        # v_loss = 0
        # v_accuracy = 0
        # v_alarm = 0
        # count = 0
        # while True:
        #     try:
        #         tmp_xentropy, tmp_correct_prediction, tmp_alarm = sess.run([cross_entropy, correct_prediction, tf.reduce_sum(tf.cast(prediction, tf.float32))])
        #         v_loss += sum(tmp_xentropy)
        #         v_accuracy += sum(tmp_correct_prediction)
        #         v_alarm += tmp_alarm
        #         count += len(tmp_xentropy)
        #     except tf.errors.OutOfRangeError:
        #         break
        # v_loss[0] /= count
        # v_accuracy[0] /= count
        # print('epoch %d validation, %d validation images, loss: %.4f, accuracy: %.4f, alarms: %d'%(epoch, count, v_loss, v_accuracy, v_alarm))
        # feed_dict = {pl_loss: v_loss[0], pl_accuracy: v_accuracy[0]}
        # validation_str = sess.run(v_summaries, feed_dict=feed_dict)
        # validation_writer.add_summary(validation_str, epoch)
        # validation_writer.flush()
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments for the model\'s trainer')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--ftrain', dest='trainable', type=bool, default=False, help='Full train (VGG)')
    parser.add_argument('--weights', dest='vgg_weights', type=str, default='vgg16_weights.npz', help='Path to the VGG\'s pretrained weights')
    parser.add_argument('--thr', dest='threshold', type=float, default=0.5, help='Model\'s detection threshold')
    parser.add_argument('--trecord', dest='train_records', type=str, default='data/train.tfrecords', help='Path to trainset tfrecords')
    parser.add_argument('--vrecord', dest='val_records', type=str, default='data/test.tfrecords', help='Path to valset tfrecords')
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--sumstep', dest='summary_step', type=int, default=50, help='Number of summary steps')
    parser.add_argument('--saveepoch', dest='save_epoch', type=int, default=10, help='Number of save epochs')
    parser.add_argument('--bs', dest='batch_size', type=int, default=20, help='Mini batch size')
    parser.add_argument('--out', dest='exp_out', type=str, default='exp', help='Path for experiment\'s outputs')
    args = parser.parse_args()
    main(args)