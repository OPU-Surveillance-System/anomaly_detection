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
from torch.utils.data import Dataset, DataLoader

import dataset as ds

def test_model(model):
    testset = ds.MiniDroneVideoDataset(args.testset, 'data', args.sequence_length)
    since = time.time()
    model.train(False)  # Set model to evaluate mode
    answer = []
    groundtruth = []
    # Iterate over data.
    dataloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=4)
    model.zero_grad()
    model.hidden = model.init_hidden()
    for i_batch, sample in enumerate(tqdm(dataloader)):
        model.zero_grad()
        model.hidden = model.init_hidden()
        inputs = Variable(sample['images'].float().cuda())[0]
        labels = Variable(sample['labels'].float().cuda())[0]
        for l in sample['labels']:
            groundtruth.append(l)
        #print(groundtruth)
        labels = np.reshape(labels, (len(labels), 1))
        # forward
        logits = model(inputs)
        probs = model.predict(logits)
        for p in probs:
            answer.append((Variable(p).data).cpu().numpy())

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return answer, groundtruth

def main(args):
    #Create experiment directory
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    #Load trained model
    model = torch.load(args.model)
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
    parser.add_argument('--sl', dest='sequence_length', type=int, default=10, help='Sequence length')
    args = parser.parse_args()
    main(args)
