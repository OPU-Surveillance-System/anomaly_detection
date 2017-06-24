"""
Train a given model according to a given dataset.
"""

import os
import tensorflow as tf
import numpy as np
from scipy import misc
import argparse
from random import shuffle
from tqdm import tqdm

import vgg16

def main(args):
    #Create output directories
    if not os.path.exists(args.exp_out):
        os.makedirs(args.exp_out)
    if not os.path.exists(args.exp_out + '/logs'):
        os.makedirs(args.exp_out + '/logs')
    if not os.path.exists(args.exp_out + '/serial'):
        os.makedirs(args.exp_out + '/serial')
    #Parse dataset
    with open(args.trainset, 'r') as f:
        content = f.read().split('\n')[:-1]
    train_set = [d.split('\t')[0:2] for d in content]
    with open(args.valset, 'r') as f:
        content = f.read().split('\n')[:-1]
    val_set = [d.split('\t')[0:2] for d in content]
    #Instantiate session
    sess = tf.InteractiveSession()
    #Instantiate model and define operations
    image = tf.placeholder(tf.float32, [None, 224, 224, 3])
    label = tf.placeholder(tf.float32, [None])
    model = vgg16.VGG16(image, args.learning_rate, args.trainable, threshold=args.threshold, weights_file=args.vgg_weights, sess=sess)
    cross_entropy = model.get_cross_entropy(image, label)
    loss_batch = model.get_loss_batch(image, label)
    correct_prediction = model.count_correct_prediction(image, label)
    accuracy_batch = model.get_accuracy_batch(image, label)
    train = model.train(image, label)
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
    sess.graph.finalize()

    #Training loop
    batch_count = 0
    for epoch in range(args.epochs):
        #Training
        step = 0
        set_size = len(train_set)
        max_step = int(set_size / args.batch_size)
        shuffle(train_set)
        while step < max_step:
            idx_start = step * args.batch_size
            idx_end = idx_start + args.batch_size
            x_batch = np.array([misc.imread(os.path.join('data', i[0])) for i in train_set[idx_start:idx_end]])
            y_batch = np.array([float(l[1]) for l in train_set[idx_start:idx_end]])
            feed_dict = {image: x_batch, label: y_batch}
            t_loss, t_accuracy, _ = sess.run([loss_batch, accuracy_batch, train], feed_dict=feed_dict)
            if step % args.summary_step == 0:
                print('epoch %d, step %d (%d images), loss: %.4f, accuracy: %.4f'%(epoch, step, (step + 1) * 20, t_loss, t_accuracy))
                feed_dict = {pl_loss: t_loss, pl_accuracy: t_accuracy}
                train_str = sess.run(t_summaries, feed_dict=feed_dict)
                train_writer.add_summary(train_str, batch_count)
                train_writer.flush()
            step += 1
            batch_count += 1
        #Save model
        if epoch % args.save_epoch == 0:
            save_path = saver.save(sess, os.path.join(args.exp_out, 'serial/model.ckpt'), global_step=epoch)
            print('Model saved in file: %s'%(save_path))
        #Validation
        # v_loss = 0
        # v_accuracy = 0
        # v_alarm = 0
        # count = 0
        # shuffle(val_set)
        # max_step = int(100 / args.batch_size) #TODO replace 100 by args.number val sample
        # step = 0
        # for step in tqdm(range(max_step)):
        #     idx_start = step * args.batch_size
        #     idx_end = idx_start + args.batch_size
        #     x_batch = np.array([misc.imread(os.path.join('data', i[0])) for i in val_set[idx_start:idx_end]])
        #     y_batch = np.array([l[1] for l in val_set[idx_start:idx_end]])
        #     feed_dict = {image: x_batch, label: y_batch}
        #     tmp_xentropy, tmp_correct_prediction, tmp_alarm = sess.run([cross_entropy, correct_prediction, tf.reduce_sum(tf.cast(prediction, tf.float32))], feed_dict=feed_dict)
        #     v_loss += sum(tmp_xentropy)
        #     v_accuracy += sum(tmp_correct_prediction)
        #     v_alarm += tmp_alarm
        #     count += len(tmp_xentropy)
        # v_loss /= count
        # v_accuracy /= count
        # print('epoch %d validation, %d validation images, loss: %.4f, accuracy: %.4f, alarms: %d'%(epoch, count, v_loss, v_accuracy, v_alarm))
        # feed_dict = {pl_loss: v_loss, pl_accuracy: v_accuracy}
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
    parser.add_argument('--tset', dest='trainset', type=str, default='data/trainset_list', help='Path to trainset')
    parser.add_argument('--vrecord', dest='valset', type=str, default='data/testset_list', help='Path to valset')
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--sumstep', dest='summary_step', type=int, default=50, help='Number of summary steps')
    parser.add_argument('--saveepoch', dest='save_epoch', type=int, default=1, help='Number of save epochs')
    parser.add_argument('--bs', dest='batch_size', type=int, default=20, help='Mini batch size')
    parser.add_argument('--out', dest='exp_out', type=str, default='exp', help='Path for experiment\'s outputs')
    args = parser.parse_args()
    main(args)
