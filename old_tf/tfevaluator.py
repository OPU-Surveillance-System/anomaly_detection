"""
"""

import os
import tensorflow as tf
import scipy
import argparse
from sklearn.metrics import roc_curve, auc, confusion_matrix
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
    preproc_label = tf.cast(parsed_features["label"], tf.float32)

    return image_tofloat, preproc_label

def main(args):
    #Create dataset
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(args.batch_size)
    #Create iterator
    iterator = dataset.make_initializable_iterator()
    image, label = iterator.get_next()
    #Instantiate session
    sess = tf.Session()
    #Instantiate model and define operations
    margs = {
        'trainable': False,
        'weights_file': None,
        'session': sess,
        'dropout': 0.0
    }
    model = vgg16.VGG16(image, label, 0.1, False, threshold=args.threshold,  margs=margs)
    logits = model.logits
    correct_prediction = model.count_correct_prediction()
    probs = model.get_probs()
    inference = model.infer()
    true_positive = tf.contrib.metrics.streaming_true_positives(inference, label)
    false_positive = tf.contrib.metrics.streaming_false_positives(inference, label)
    false_negative = tf.contrib.metrics.streaming_false_negatives(inference, label)
    true_negative = tf.contrib.metrics.streaming_true_negatives(inference, label)
    op_auc = tf.metrics.auc(label, probs, num_thresholds=200, curve='ROC', name=None)
    #Init variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    #Restore trained model
    saver = tf.train.Saver()
    saver.restore(sess, args.model_path)
    #Evaluation loop
    testing_filenames = [args.test_records]
    sess.run(iterator.initializer, feed_dict={filenames: testing_filenames})
    step = 0
    model_responses = []
    groundtruths = []
    accuracy = 0
    count = 0
    print('Computing logits on the testset...')
    while True:
        try:
            score, detection, answer, tp, fp, fn, tn, tmp_correct_prediction, tfauc = sess.run([logits, probs, label, true_positive, false_positive, false_negative, true_negative, correct_prediction, op_auc])
            model_responses += list(detection)
            groundtruths += list(answer)
            accuracy += sum(tmp_correct_prediction)
            count += len(tmp_correct_prediction)
            print(score, answer)
        except tf.errors.OutOfRangeError:
            print('Evaluation complete')
            break
    accuracy /= count
    print('COUNT %d'%(count))
    print('%d abnormal frames within %d frames'%(sum(groundtruths), len(groundtruths)))
    print('%f true positives, %f false positives, %f false negatives, %f true negatives, accuracy: %f, TF AUC: %f'%(tp[1], fp[1], fn[1], tn[1], accuracy, tfauc[1]))
    #AUC measure
    fpr, tpr, thresholds = roc_curve(groundtruths, model_responses)
    roc_auc = auc(fpr, tpr)
    print("AUC score %f"%(roc_auc))
    #Plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    for thr in range(0, len(fpr), 200):
        plt.text(fpr[thr], tpr[thr], thresholds[thr])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.exp_out, 'roc.svg'), format='svg')
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments for the model\'s trainer')
    parser.add_argument('--thr', dest='threshold', type=float, default=0.5, help='Model\'s detection threshold')
    parser.add_argument('--trecord', dest='test_records', type=str, default='data/testset.tfrecord', help='Path to testset tfrecords')
    parser.add_argument('--bs', dest='batch_size', type=int, default=20, help='Mini batch size')
    parser.add_argument('--out', dest='exp_out', type=str, default='exp', help='Path for experiment\'s outputs')
    parser.add_argument('--model', dest='model_path', type=str, default='exp/serial/model.ckpt', help='Path to the trained model')
    args = parser.parse_args()
    main(args)
