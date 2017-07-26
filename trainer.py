"""
Train a given model according to a given dataset.
"""

import os
import tensorflow as tf
from tensorflow.python.framework import ops
import argparse
from scipy import misc
from sklearn.metrics import roc_curve, auc, confusion_matrix

import win_vgg16

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
    preproc_label = tf.cast(parsed_features["label"], tf.float32)

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
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(args.batch_size)
    #Create iterator
    iterator = dataset.make_initializable_iterator()
    image, label = iterator.get_next()
    #Instantiate session
    sess = tf.Session()
    #Instantiate model and define operations
    is_training = tf.placeholder(tf.bool, name='pl_istraining')
    margs = {
        'trainable': args.trainable,
        'weights_file': args.vgg_weights,
        'session': sess,
        'dropout': args.dpr
    }
    model = win_vgg16.VGG16(image, label, args.learning_rate, is_training, threshold=args.threshold,  margs=margs)
    #model = vgg16.VGG16(image, label, args.learning_rate, args.trainable, threshold=args.threshold, weights_file=args.vgg_weights, sess=sess)
    cross_entropy = model.get_cross_entropy()
    loss_batch = model.get_loss_batch()
    correct_prediction = model.count_correct_prediction()
    accuracy_batch = model.get_accuracy_batch()
    train, learning_rate = model.train()
    probs = model.get_probs()
    #Create summaries
    pl_loss = tf.placeholder(tf.float32, name='loss_placeholder')
    pl_accuracy = tf.placeholder(tf.float32, name='accuracy_placeholder')
    pl_auc = tf.placeholder(tf.float32, name='auc_placeholder')
    pl_lr = tf.placeholder(tf.float32, name='learning_rate_placeholder')
    with tf.variable_scope("train_set"):
        t_loss_summary = tf.summary.scalar(tensor=pl_loss, name='loss')
        t_accuracy_summary = tf.summary.scalar(tensor=pl_accuracy, name='accuracy')
        t_lr_summary = tf.summary.scalar(tensor=pl_lr, name='learning_rate')
        t_summaries = tf.summary.merge([t_loss_summary, t_accuracy_summary, t_lr_summary])
    with tf.variable_scope("validation_set"):
        v_loss_summary = tf.summary.scalar(tensor=pl_loss, name='loss')
        v_accuracy_summary = tf.summary.scalar(tensor=pl_accuracy, name='accuracy')
        v_auc_summary = tf.summary.scalar(tensor=pl_auc, name='auc')
        v_summaries = tf.summary.merge([v_loss_summary, v_accuracy_summary, v_auc_summary])
    train_writer = tf.summary.FileWriter(os.path.join(args.exp_out, 'logs/train'), sess.graph)
    validation_writer = tf.summary.FileWriter(os.path.join(args.exp_out, 'logs/validation'), sess.graph)
    #Create saver
    saver = tf.train.Saver(max_to_keep=0)
    #Init variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print(model.parameters)
    print(len(model.parameters))

    #Training loop
    batch_count = 0
    for epoch in range(args.epochs):
        #Training
        training_filenames = args.train_records.split(',')
        sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
        step = 0
        while True:
            feed_dict = {is_training: True}
            try:
                t_loss, t_accuracy, _, lr, logits, gt, det = sess.run([loss_batch, accuracy_batch, train, learning_rate, model.logits, label, probs], feed_dict=feed_dict)
                #t_loss, t_accuracy, _, logits, gt, det = sess.run([loss_batch, accuracy_batch, train, model.logits, label, probs], feed_dict=feed_dict)
                if step % args.summary_step == 0:
                    print('epoch %d, step %d (%d images), loss: %.4f, accuracy: %.4f'%(epoch, step, (step + 1) * args.batch_size, t_loss, t_accuracy))
                    print(logits[0:10], gt[0:10], sum(gt), sum(det))
                    feed_dict = {pl_loss: t_loss, pl_accuracy: t_accuracy, pl_lr: lr}
                    train_str = sess.run(t_summaries, feed_dict=feed_dict)
                    train_writer.add_summary(train_str, batch_count)
                    train_writer.flush()
                step += 1
                batch_count += 1
            except tf.errors.OutOfRangeError:
                print('Epoch %d complete'%(epoch))
                break
        #Save model
        #if epoch % args.save_epoch == 0:
        save_path = saver.save(sess, os.path.join(args.exp_out, 'serial/model.ckpt'), global_step=epoch)
        print('Model saved in file: %s'%(save_path))
        #Validation
        validation_filenames = [args.val_records]
        feed_dict = {is_training: False}
        sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
        v_loss = 0
        v_accuracy = 0
        count = 0
        model_responses = []
        groundtruths = []
        while True:
            try:
                tmp_xentropy, tmp_correct_prediction, logits, gt, ans = sess.run([cross_entropy, correct_prediction, model.logits, label, probs], feed_dict=feed_dict)
                print(logits, gt)
                v_loss += sum(tmp_xentropy)
                v_accuracy += sum(tmp_correct_prediction)
                count += len(tmp_xentropy)
                model_responses += list(ans)
                groundtruths += list(gt)
            except tf.errors.OutOfRangeError:
                break
        fpr, tpr, thresholds = roc_curve(groundtruths, model_responses)
        roc_auc = auc(fpr, tpr)
        v_loss /= count
        v_accuracy /= count
        print('epoch %d validation, %d validation images, loss: %.4f, accuracy: %.4f'%(epoch, count, v_loss, v_accuracy))
        feed_dict = {pl_loss: v_loss, pl_accuracy: v_accuracy, pl_auc: roc_auc}
        validation_str = sess.run(v_summaries, feed_dict=feed_dict)
        validation_writer.add_summary(validation_str, epoch)
        validation_writer.flush()
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments for the model\'s trainer')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.000001, help='Learning rate')
    parser.add_argument('--ftrain', dest='trainable', type=bool, default=False, help='Full train (VGG)')
    parser.add_argument('--weights', dest='vgg_weights', type=str, default='vgg16_weights.npz', help='Path to the VGG\'s pretrained weights')
    parser.add_argument('--thr', dest='threshold', type=float, default=0.5, help='Model\'s detection threshold')
    parser.add_argument('--trecord', dest='train_records', type=str, default='data/augtrainset.tfrecord', help='Path to trainset tfrecords')
    parser.add_argument('--vrecord', dest='val_records', type=str, default='data/valset.tfrecord', help='Path to valset tfrecords')
    parser.add_argument('--epochs', dest='epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--sumstep', dest='summary_step', type=int, default=50, help='Number of summary steps')
    parser.add_argument('--saveepoch', dest='save_epoch', type=int, default=10, help='Number of save epochs')
    parser.add_argument('--bs', dest='batch_size', type=int, default=20, help='Mini batch size')
    parser.add_argument('--out', dest='exp_out', type=str, default='exp', help='Path for experiment\'s outputs')
    parser.add_argument('--dpr', dest='dpr', type=float, default=0.5, help='Dropout probability')
    args = parser.parse_args()
    main(args)
