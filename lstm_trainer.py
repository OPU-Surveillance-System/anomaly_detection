from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import time
import copy
import os
from scipy import misc
import argparse
from random import shuffle
from importlib import import_module

import plot as plt
import data_augmentation as da
import dataset as ds

def train_model(model, loss_function, optimizer):
    """
    """

    with open(args.trainset, 'r') as f:
        trainset = f.read().split('\n')[:-1]
    trainset = [(t.split('\t')[0:10], t.split('\t')[10:]) for t in trainset]
    with open(args.valset, 'r') as f:
        valset = f.read().split('\n')[:-1]
    valset = [(v.split('\t')[0:10], v.split('\t')[10:]) for v in valset]
    trainset = ds.MiniDroneVideoDataset(args.trainset,
                                        'data',
                                        10,
                                        transform=transforms.Compose([
                                               ds.RandomCrop((160, 160)),
                                               ds.RandomFlip(),
                                               ds.Dropout(0.2)
                                           ]))
    valset = ds.MiniDroneVideoDataset(args.valset, 'data', 10)
    dsets = {'training': trainset, 'validation': valset}
    phase = list(dsets.keys())
    dset_sizes = {p: len(dsets[p]) for p in phase}
    dset_sizes['training'] = int(dset_sizes['training'] / 2)
    trainepoch = 0
    accumulated_patience = 0
    best_loss = float('inf')
    best_loss_acc = 0
    best_model = copy.deepcopy(model)
    best_trainepoch = 0
    hist = {'training':{'loss':[], 'accuracy':[]}, 'validation':{'loss':[], 'accuracy':[]}}
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
            nb_frames = 0
            dataloader = DataLoader(dsets[p], batch_size=1, shuffle=True, num_workers=4)
            for i_batch, sample in enumerate(dataloader):
                #Initialize model's gradient and LSTM state
                model.zero_grad()
                model.hidden = model.init_hidden()
                #Fetch sequence frames and labels
                # inputs = np.array([misc.imread(os.path.join(dsets[p][step][0][i])) for i in range(len(dsets[p][step][0]))])
                # if p == 'training' and args.augdata == 1:
                #     inputs = da.augment_batch(inputs)
                # inputs = np.transpose(inputs, (0, 3, 1, 2))
                # labels = np.array([dsets[p][step][1][i] for i in range(len(dsets[p][step][1]))], dtype=np.float).reshape((len(dsets[p][step][1]), 1))
                # #Convert to cuda tensor
                # inputs = Variable(torch.from_numpy(inputs).float().cuda())
                # labels = Variable(torch.from_numpy(labels).float().cuda())
                inputs = sample['images']
                labels = sample['labels']
                #Forward
                logits = model(inputs)
                probs = model.predict(logits)
                loss = loss_function(logits, labels)
                if p == 'training': #Backpropagation
                    loss.backward()
                    optimizer.step()
                running_loss += loss.data[0]
                batch_loss = loss.data[0]
                batch_corrects = torch.sum(probs == labels.data.long())
                running_corrects += torch.sum(probs == labels.data.long())
                nb_frames += len(probs)
                if p == 'training' and step % 1000 == 0:
                    print('{} : step {} -- Loss: {} Acc: {}'.format(p, step, batch_loss / len(inputs), batch_corrects / len(inputs)))
            epoch_loss = running_loss / dset_sizes[p]
            epoch_acc = running_corrects / nb_frames
            hist[p]['loss'].append(epoch_loss)
            hist[p]['accuracy'].append(epoch_acc)
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
                if args.plot == 1:
                    plt.plot_history(hist, args.directory)
        trainepoch += 1
    print()
    time_elapsed = time.time() - t_start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation loss: {}, accuracy: {}, epoch: {}'.format(best_loss, best_loss_acc, best_trainepoch))

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
    parser.add_argument('--sl', dest='sequence_length', type=int, default=10, help='Sequence length')
    args, unknown = parser.parse_known_args()
    margs = {u.split('=')[0][2:]:u.split('=')[1] for u in unknown}
    print('arguments passed to the model: {}'.format(margs))
    main(args, margs)
