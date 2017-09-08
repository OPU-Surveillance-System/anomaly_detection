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
from importlib import import_module

import plot as plt
import data_augmentation as da

def train_model(model, loss_function, optimizer):
    """
    """

    with open(args.trainset, 'r') as f:
        trainset = f.read().split('\n')[:-1]
    #TODO: Sequence trainset
    trainset = [([t.split('\t')[0] for i in range(10)], [t.split('\t')[1] for i in range(10)]) for t in trainset]
    with open(args.valset, 'r') as f:
        valset = f.read().split('\n')[:-1]
    #TODO: Sequence trainset
    valset = [([v.split('\t')[0] for i in range(10)], [v.split('\t')[1] for i in range(10)]) for v in valset]
    dsets = {'training': trainset, 'validation': valset}
    phase = list(dsets.keys())
    dset_sizes = {p: len(dsets[p]) for p in phase}
    trainepoch = 0
    accumulated_patience = 0
    best_loss = float('inf')
    best_loss_acc = 0
    best_model = copy.deepcopy(model)
    best_trainepoch = 0
    t_start = time.time()
    while accumulated_patience < args.max_patience:
        print('-' * 10)
        print('Epoch {} (patience: {}/{})'.format(trainepoch, accumulated_patience, args.max_patience))
        for p in phase:
            if p == 'training':
                model.train(True)
            else:
                model.train(False)
            running_loss = 0
            running_corrects = 0
            shuffle(dsets[p])
            for step in range(int(dset_sizes[p] / 100)):
                #Initialize model's gradient and LSTM state
                model.zero_grad()
                model.hidden = model.init_hidden()
                #Fetch sequence frames and labels
                inputs = np.array([misc.imread(os.path.join('data', dsets[p][step][0][i] + '.png')) for i in range(10)], dtype=np.float)
                inputs = np.transpose(inputs, (0, 3, 1, 2))
                if p == 'training' and args.augdata == 1:
                    #TODO: Apply the same augmentation to each element
                    inputs = da.augment_batch(inputs)
                labels = np.array([dsets[p][step][1][i] for i in range(10)], dtype=np.float).reshape((10, 1))
                #Convert to cuda tensor
                inputs = Variable(torch.from_numpy(inputs).float().cuda())
                labels = Variable(torch.from_numpy(labels).float().cuda())
                #Forward
                logits = model(inputs)
                probs = model.predict(logits)
                loss = loss_function(logits, labels)
                if p == 'training': #Backpropagation
                    loss.backward()
                    optimizer.step()
                running_loss += loss.data[0]
                running_corrects += torch.sum(probs == labels.data.long())
            epoch_loss = running_loss / dset_sizes[p]
            epoch_acc = running_corrects / dset_sizes[p] * 10
            print('{} -- Loss: {} Acc: {}'.format(p, epoch_loss, epoch_acc))
            if p == 'validation':
                if epoch_loss < best_loss:
                    accumulated_patience = 0
                    best_model = copy.deepcopy(model)
                    best_trainepoch = trainepoch
                    best_loss = epoch_loss
                    best_loss_acc = epoch_acc
                else:
                    accumulated_patience += 1
        trainepoch += 1

    return best_model, best_trainepoch

def main(args, margs):
    """
    1. Create a directory for the experiment results.
    2. Dynamically instantiate the specified model and train it.
    3. Save the trained model in the created directory.
    """

    #1.
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    #2.
    model_import = import_module('.'.join(args.model.split('.')[0:2]))
    model_class = getattr(model_import, args.model.split('.')[2])
    model = model_class(margs)
    model = model.cuda()
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.trainable_parameters, args.learning_rate)
    trained_model, best_trainepoch = train_model(model, loss_function, optimizer)
    #3.
    torch.save(trained_model, os.path.join(args.directory, 'trained_model'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--m', dest='model', type=str, default='models.vgg16lstm.VGG16LSTM', help='')
    parser.add_argument('--bs', dest='batch_size', type=int, default=1, help='')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--p', dest='max_patience', type=int, default=10, help='')
    parser.add_argument('--tr', dest='trainset', type=str, default='data/trainset_labels', help='')
    parser.add_argument('--val', dest='valset', type=str, default='data/valset_labels', help='')
    parser.add_argument('--dir', dest='directory', type=str, default='experiment', help='')
    parser.add_argument('--da', dest='augdata', type=int, default=1, help='')
    parser.add_argument('--plot', dest='plot', type=int, default=1, help='')
    args, unknown = parser.parse_known_args()
    margs = {u.split('=')[0][2:]:u.split('=')[1] for u in unknown}
    print('arguments passed to the model: {}'.format(margs))
    main(args, margs)
