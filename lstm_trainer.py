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
import argparse
from importlib import import_module
from tqdm import tqdm

import plot as plt
import dataset as ds

def train_model(model, loss_function, optimizer):
    """
    """

    tsfm = ds.Normalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    da = tsfm
    if args.augdata == 1:
        da = [da] + [ds.RandomCrop((160, 160)), ds.RandomFlip(), ds.Dropout(0.2)]
    trainset = ds.MiniDroneVideoDataset(args.trainset, 'data', args.sequence_length, args.stride, transform=transforms.Compose(da))
    valset = ds.MiniDroneVideoDataset(args.valset, 'data', args.sequence_length, args.stride, transform=tsfm)
    dsets = {'training': trainset, 'validation': valset}
    phase = list(dsets.keys())
    dset_sizes = {p: (len(dsets[p]) * args.sequence_length) for p in phase}
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
            dataloader = DataLoader(dsets[p], batch_size=1, shuffle=True, num_workers=4)
            for i_batch, sample in enumerate(tqdm(dataloader)):
                #Initialize model's gradient and LSTM state
                model.zero_grad()
                model.hidden = model.init_hidden()
                # #Convert to cuda tensor
                inputs = Variable(sample['images'].float().cuda())[0]
                labels = Variable(sample['labels'].float().cuda())[0]
                #Forward
                logits = model(inputs)
                probs = model.threshold(logits)
                loss = loss_function(logits, labels)
                if p == 'training': #Backpropagation
                    loss.backward()
                    optimizer.step()
                running_loss += loss.data[0]
                running_corrects += torch.sum(probs == labels.data.long())
            epoch_loss = running_loss / dset_sizes[p]
            epoch_acc = running_corrects / dset_sizes[p]
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
    parser.add_argument('--str', dest='stride', type=int, default=10, help='Sliding window stride')
    parser.add_argument('--stop', dest='stop', type=int, default=1, help='')
    args, unknown = parser.parse_known_args()
    margs = {u.split('=')[0][2:]:u.split('=')[1] for u in unknown}
    print('arguments passed to the model: {}'.format(margs))
    main(args, margs)
