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

def train_model(model_to_train):
    model = model_to_train.model
    criterion = model_to_train.criterion
    optimizer = model_to_train.optimizer
    #Read data
    with open(args.trainset, 'r') as f:
        trainset = f.read().split('\n')[:-1]
    trainset = [(c.split('\t')[0], int(c.split('\t')[1])) for c in trainset]
    with open(args.valset, 'r') as f:
        valset = f.read().split('\n')[:-1]
    valset = [(c.split('\t')[0], int(c.split('\t')[1])) for c in valset]
    dsets = {'training': trainset, 'validation': valset}
    dset_sizes = {x: len(dsets[x]) for x in ['training', 'validation']}
    #Local variables
    phase = ['training', 'validation']
    m = nn.Sigmoid()
    trainepoch = 0
    accumulated_patience = 0
    best_loss = float('inf')
    best_model = copy.deepcopy(model)
    best_trainepoch = 0
    max_trainstep = int(len(trainset) / args.batch_size)
    max_valstep = int(len(valset) / args.batch_size)
    hist = {'training':{'loss':[], 'accuracy':[]}, 'validation':{'loss':[], 'accuracy':[]}}
    #Train loop
    while accumulated_patience < args.max_patience:
        print('Epoch {}'.format(trainepoch))
        print('-' * 10)
        for p in phase:
            if p == 'training':
                model.train(True)
                optimizer.zero_grad() # zero the parameter gradients
            else:
                model.train(False)
            running_loss = 0
            running_corrects = 0
            shuffle(dsets[p])
            maxstep = int(len(dsets[p]) / args.batch_size)
            for step in range(maxstep):
                idx_start = step * args.batch_size
                idx_end = idx_start + args.batch_size
                inputs = np.array([misc.imread(os.path.join('data', dsets[p][i][0] + '.png')) for i in range(idx_start, idx_end)])
                if p == 'training' and args.augdata:
                    inputs = augment_batch(inputs)
                inputs = np.transpose(inputs, (0, 3, 1, 2))
                labels = np.array([dsets[p][i][1] for i in range(idx_start, idx_end)])
                labels = np.reshape(labels, (args.batch_size, 1))
                #convert to cuda tensor
                inputs, labels = torch.from_numpy(inputs).float(), torch.from_numpy(labels).float()
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                #forward
                outputs = model(inputs)
                preds = model.predict(outputs.data)
                loss = criterion(outputs, labels)
                if p == 'training':
                    # backward + optimize
                    loss.backward()
                    optimizer.step()
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data.long())
            epoch_loss = running_loss / len(dsets[p])
            epoch_acc = running_corrects / len(dsets[p])
            hist[p]['loss'].append(epoch_loss)
            hist[p]['accuracy'].append(epoch_acc)
            print('{} -- Loss: {} Acc: {}'.format(p, epoch_loss, epoch_acc))
            trainepoch+= 1
            if p == 'validation':
                if epoch_loss < best_loss:
                    accumulated_patience = 0
                    best_model = copy.deepcopy(model)
                    best_trainepoch = i
                    best_loss = val_loss
                else:
                    accumulated_patience += 1
                if args.plot_hist:
                    plot_history(hist)
    print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return best_model, best_trainepoch

def main(args, margs):
    #Create experiment directory
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    model_import = import_module('.'.join(args.model.split('.')[0:1]))
    model_class = getattr(model_import, args.model.split('.')[1])
    model = model_class(margs)
    trained_model, best_trainepoch = train_model(model)
    model.model = deepcopy(trained_model)
    torch.save(model.model, os.path.join(args.directory, 'best_loss_model'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments for the model\'s trainer')
    parser.add_argument('--m', dest='model', type=str, default='vgg16extractor.VGG16Extractor')
    parser.add_argument('--bs', dest='batch_size', type=int, default=40, help='Batch size')
    parser.add_argument('--p', dest='max_patience', type=int, default=10, help='')
    parser.add_argument('--tr', dest='trainset', type=str, default='data/trainset_labels', help='Path to the trainset summary')
    parser.add_argument('--val', dest='valset', type=str, default='data/valset_labels', help='Path to the valset summary')
    parser.add_argument('--dir', dest='directory', type=str, default='experiment', help='Path to a directory for saving results')
    parser.add_argument('--da', dest='augdata', type=bool, default=False, help='Whether to activate data augmentation pipeline or not during training')
    args, unknown = parser.parse_known_args()
    margs = {u.split('=')[0][2:]:u.split('=')[1] for u in unknown}
    main(args, margs)
