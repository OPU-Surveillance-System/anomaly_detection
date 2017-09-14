from __future__ import print_function, division
import torch
from torch.autograd import Variable
import numpy as np
import time
import os
import argparse
from sklearn import metrics
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import dataset as ds
import plot as plt

def test_model(model):
    tsfm = ds.Normalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    testset = ds.MiniDroneVideoDataset(args.testset, 'data', args.sequence_length, transform=tsfm)
    since = time.time()
    model.train(False)  # Set model to evaluate mode
    answer = []
    groundtruth = []
    names = []
    # Iterate over data.
    dataloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=4)
    model.zero_grad()
    model.hidden = model.init_hidden()
    running_corrects = 0
    for i_batch, sample in enumerate(tqdm(dataloader)):
        model.zero_grad()
        model.hidden = model.init_hidden()
        inputs = Variable(sample['images'].float().cuda())[0]
        labels = Variable(sample['labels'].float().cuda())[0]
        # forward
        logits = model(inputs)
        probs = model.predict(logits)
        detection = model.threshold(logits)
        running_corrects += torch.sum(detection == labels.data.long())
        groundtruth.append(labels.data.cpu().numpy())
        answer.append(Variable(probs).data.cpu().numpy())
        names.append(sample['names'])
    accuracy = running_corrects / (len(testset) * args.sequence_length)
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return answer, groundtruth, accuracy, names

def main(args):
    #Create experiment directory
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    #Load trained model
    model = torch.load(args.model)
    model = model.cuda()
    answer, groundtruth, accuracy, names = test_model(model)
    answer, groundtruth = np.array(answer), np.array(groundtruth)
    answer = answer.reshape(answer.shape[0] * answer.shape[1])
    groundtruth = groundtruth.reshape(groundtruth.shape[0] * groundtruth.shape[1])
    keys = ['tp', 'tn', 'fp', 'fn']
    named = {k:[] for k in keys}
    for i in range(len(answer)):
        if answer[i] == 1 and groundtruth == 1:
            named['tp'].append(names[i])
        elif answer[i] == 0 and groundtruth == 0:
            named['tn'].append(names[i])
        elif answer[i] == 1 and groundtruth == 0:
            named['fp'].append(names[i])
        elif answer[i] == 0 and groundtruth == 1:
            named['fn'].append(names[i])
    for k in keys:
        with open(k, 'w') as f:
            for elt in named[k]:
                f.write('{}\n'.format(elt))
    print('Accuracy @{}: {}'.format(model.margs['thr'], accuracy))
    fpr, tpr, thresholds = metrics.roc_curve(groundtruth, answer)
    auc = metrics.auc(fpr, tpr)
    plt.plot_auc(auc, fpr, tpr, args.directory, os.path.basename(args.model))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments for the model\'s trainer')
    parser.add_argument('--bs', dest='batch_size', type=int, default=40, help='Batch size')
    parser.add_argument('--te', dest='testset', type=str, default='data/testset_labels', help='Path to the testset summary')
    parser.add_argument('--dir', dest='directory', type=str, default='experiment', help='Path to a directory for saving results')
    parser.add_argument('--m', dest='model', type=str, default='modelsave', help='Path to the serialized model')
    parser.add_argument('--sl', dest='sequence_length', type=int, default=10, help='Sequence length')
    args = parser.parse_args()
    main(args)
