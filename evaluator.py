from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
import os
from scipy import misc
import argparse
from random import shuffle
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm

use_gpu = torch.cuda.is_available()

def test_model(model):
    with open(args.testset, 'r') as f:
        trainset = f.read().split('\n')[:-1]
    testset = [(c.split('\t')[0], int(c.split('\t')[1])) for c in trainset]
    since = time.time()
    model.train(False)  # Set model to evaluate mode
    step = 0
    max_step = int(len(testset) / args.batch_size)
    answer = []
    groundtruth = []

    # Iterate over data.
    for step in tqdm(range(max_step)):
        # get the inputs
        idx_start = step * args.batch_size
        idx_end = idx_start + args.batch_size
        inputs = np.array([misc.imread(os.path.join('data', testset[i][0] + '.png')) for i in range(idx_start, idx_end)])
        inputs = np.transpose(inputs, (0, 3, 1, 2))
        labels = np.array([testset[i][1] for i in range(idx_start, idx_end)])
        for l in labels:
            groundtruth.append(l)
        labels = np.reshape(labels, (args.batch_size, 1))
        #convert to tensor
        inputs, labels = torch.from_numpy(inputs).float(), torch.from_numpy(labels).float()
        # wrap them in Variable
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        # forward
        outputs = model(inputs)
        preds = torch.sigmoid(outputs.data)
        for p in preds:
            answer.append((Variable(p).data).cpu().numpy())
        #print('Processed {} images'.format(len(inputs)))

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return answer, groundtruth

def main(args):
    #Create experiment directory
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    #Load trained model
    model = torch.load(args.model)
    if use_gpu:
        model = model.cuda()
    answer, groundtruth = test_model(model)
    answer = np.array(answer).reshape(len(answer))
    groundtruth = np.array(groundtruth).reshape(len(groundtruth))
    fpr, tpr, thresholds = metrics.roc_curve(groundtruth, answer)
    auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.directory, os.path.basename(args.model) + '_roc.svg'), format='svg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments for the model\'s trainer')
    parser.add_argument('--bs', dest='batch_size', type=int, default=40, help='Batch size')
    parser.add_argument('--te', dest='testset', type=str, default='data/testset_labels', help='Path to the testset summary')
    parser.add_argument('--dir', dest='directory', type=str, default='experiment', help='Path to a directory for saving results')
    parser.add_argument('--m', dest='model', type=str, default='modelsave', help='Path to the serialized model')
    args = parser.parse_args()
    main(args)
