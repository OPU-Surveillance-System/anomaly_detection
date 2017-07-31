"""
Train a given model according to a given dataset.
"""

import os
import tensorflow as tf
from tensorflow.python.framework import ops
import argparse
from scipy import misc

import lstmvgg16

def main(args):
    #Create output directories
    if not os.path.exists(args.exp_out):
        os.makedirs(args.exp_out)
    if not os.path.exists(args.exp_out + '/logs'):
        os.makedirs(args.exp_out + '/logs')
    if not os.path.exists(args.exp_out + '/serial'):
        os.makedirs(args.exp_out + '/serial')
    #Create dataset
    with open(args.trainset, 'r') as f:
        trainset = f.read().split('\n')[:-1]
    trainset = [c.split('\t') for c in trainset]
    with open(args.valset, 'r') as f:
        valset = f.read().split('\n')[:-1]
    valset = [c.split('\t') for c in valset]
    #Instantiate session
    sess = tf.Session()
    #Instantiate model and define operations
    image = tf.placeholder(tf.float32, name='pl_input')
    label = tf.placeholder(tf.float32, name='pl_label')
    init_state = tf.placeholder(tf.float32, name='pl_init_state')
    is_training = tf.placeholder(tf.bool, name='pl_init_state')
    margs = {
        'trainable': args.trainable,
        'weights file': args.vgg_weights,
        'session': sess,
        'dropout': args.dpr,
        'init state': init_state,
        'state size': args.state_size,
        'lstm num layers': args.lstm_num_layers
    }
    model = lstmvgg16.LSTMVGG16(image, label, args.learning_rate, is_training, threshold=args.threshold,  margs=margs)
    cross_entropy = model.get_cross_entropy()
    loss_batch = model.get_loss_batch()
    correct_prediction = model.count_correct_prediction()
    accuracy_batch = model.get_accuracy_batch()
    train, learning_rate = model.train()
    probs = model.get_probs()
    #Create summaries
    pl_loss = tf.placeholder(tf.float32, name='loss_placeholder')
    pl_accuracy = tf.placeholder(tf.float32, name='accuracy_placeholder')
    pl_lr = tf.placeholder(tf.float32, name='learning_rate_placeholder')
    with tf.variable_scope("train_set"):
        t_loss_summary = tf.summary.scalar(tensor=pl_loss, name='loss')
        t_accuracy_summary = tf.summary.scalar(tensor=pl_accuracy, name='accuracy')
        t_lr_summary = tf.summary.scalar(tensor=pl_lr, name='learning_rate')
        t_summaries = tf.summary.merge([t_loss_summary, t_accuracy_summary, t_lr_summary])
    with tf.variable_scope("validation_set"):
        v_loss_summary = tf.summary.scalar(tensor=pl_loss, name='loss')
        v_accuracy_summary = tf.summary.scalar(tensor=pl_accuracy, name='accuracy')
        v_summaries = tf.summary.merge([v_loss_summary, v_accuracy_summary])
    train_writer = tf.summary.FileWriter(os.path.join(args.exp_out, 'logs/train'), sess.graph)
    validation_writer = tf.summary.FileWriter(os.path.join(args.exp_out, 'logs/validation'), sess.graph)
    #Create saver
    saver = tf.train.Saver(max_to_keep=10)
    #Init variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print(model.parameters)
    print(len(model.parameters))

    #Training loop
    batch_count = 0
    for epoch in range(args.epochs):
        #Training
        step = 0
        max_step = len(trainset) // args.bs
        while step < max_step:
            idx_start = step * args.bs
            idx_end = idx_start + args.bs
            img = [[misc.imread(trainset[i][f]) for f in range(args.sliding_window_len)] for i in range(idx_start, idx_end)]
            lbl = [int(trainset[i][-1]) for i in range(idx_start, idx_end)]
            _current_state = np.zeros((args.lstm_num_layers, 2, args.bs, args.state_size))
            feed_dict = {image:img, label:lbl, init_state:_current_state}
            t_loss, t_accuracy, _, lr, logits, gt = sess.run([loss_batch, accuracy_batch, train, learning_rate, model.logits, label], feed_dict=feed_dict)
            if step % args.summary_step == 0:
                print('epoch %d, step %d (%d images), loss: %.4f, accuracy: %.4f'%(epoch, step, (step + 1) * args.batch_size, t_loss, t_accuracy))
                print(logits[0:10], gt[0:10], sum(gt), sum(det))
                feed_dict = {pl_loss: t_loss, pl_accuracy: t_accuracy, pl_lr: lr}
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
        v_loss = 0
        v_accuracy = 0
        count = 0
        step = 0
        max_step = len(valset) // args.bs
        while step < max_step:
            idx_start = step * args.bs
            idx_end = idx_start + args.bs
            img = [[misc.imread(valset[i][f]) for f in range(args.sliding_window_len)] for i in range(idx_start, idx_end)]
            lbl = [int(valset[i][-1]) for i in range(idx_start, idx_end)]
            _current_state = np.zeros((args.lstm_num_layers, 2, args.bs, args.state_size))
            feed_dict = {image:img, label:lbl, init_state:_current_state}
            tmp_xentropy, tmp_correct_prediction, logits, gt = sess.run([cross_entropy, correct_prediction, model.logits, label], feed_dict=feed_dict)
            v_loss += sum(tmp_xentropy)
            v_accuracy += sum(tmp_correct_prediction)
            count += len(tmp_xentropy)
            print(logits, gt)
        v_loss /= count
        v_accuracy /= count
        print('epoch %d validation, %d validation images, loss: %.4f, accuracy: %.4f'%(epoch, count, v_loss, v_accuracy))
        feed_dict = {pl_loss: v_loss, pl_accuracy: v_accuracy}
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
    parser.add_argument('--trainset', dest='trainset', type=str, default='data/augmentatedtrainset', help='Path to the trainset summary')
    parser.add_argument('--valset', dest='valset', type=str, default='data/valset', help='Path to the valset summary')
    parser.add_argument('--epochs', dest='epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--sumstep', dest='summary_step', type=int, default=50, help='Number of summary steps')
    parser.add_argument('--saveepoch', dest='save_epoch', type=int, default=10, help='Number of save epochs')
    parser.add_argument('--bs', dest='batch_size', type=int, default=20, help='Mini batch size')
    parser.add_argument('--out', dest='exp_out', type=str, default='exp', help='Path for experiment\'s outputs')
    parser.add_argument('--dpr', dest='dpr', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--statesize', dest='state_size', type=int, default=4, help='LSTM state size')
    parser.add_argument('--lstml', dest='lstm_num_layers', type=int, default=1, help='Number of stacked LSTM')
    parser.add_argument('--swl', dest='sliding_window_len', type=int, default=10, help='Length of the sliding window')
    args = parser.parse_args()
    main(args)
