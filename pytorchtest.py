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

use_gpu = torch.cuda.is_available()

with open('data/augmentated_trainset', 'r') as f:
    trainset = f.read().split('\n')[:-1]
trainset = [(c.split('\t')[0], int(c.split('\t')[1])) for c in trainset]
with open('data/valset_labels', 'r') as f:
    valset = f.read().split('\n')[:-1]
valset = [(c.split('\t')[0], int(c.split('\t')[1])) for c in valset]
dsets = {'train': trainset, 'val': valset}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}

def train_model(model, criterion, optimizer, lr_scheduler):
    since = time.time()

    best_model = model
    best_acc = 0.0
    summaries = {'train':{'loss':[], 'accuracy':[]}, 'val':{'loss':[], 'accuracy':[]}}

    fig, axes = plt.subplots(nrows=1, ncols=2)
    color = {'train':'blue', 'val':'red'}

    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            step = 0
            max_step = int(dset_sizes[phase] / args.batch_size)

            # Iterate over data.
            while step < int(max_step / 2):
                # get the inputs
                idx_start = step * args.batch_size
                idx_end = idx_start + args.batch_size
                shuffle(dsets[phase])
                inputs = np.array([misc.imread(os.path.join('data', dsets[phase][i][0] + '.png')) for i in range(idx_start, idx_end)])
                inputs = np.transpose(inputs, (0, 3, 1, 2))
                labels = np.array([dsets[phase][i][1] for i in range(idx_start, idx_end)])
                labels = np.reshape(labels, (args.batch_size, 1))
                #convert to tensor
                inputs, labels = torch.from_numpy(inputs).float(), torch.from_numpy(labels).float()
                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds.int() == labels.data.int())
                # next step
                step += 1
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]
            summaries[phase]['loss'].append(epoch_loss)
            summaries[phase]['accuracy'].append(epoch_acc)
            lineloss = axes[0].plot(list(range(epoch + 1)), summaries[phase]['loss'], color=color[phase], label='%s loss'%(phase))
            lineacc = axes[1].plot(list(range(epoch + 1)), summaries[phase]['accuracy'], color=color[phase], label='%s accuracy'%(phase))

            print('{} Loss: {} Acc: {}, some outputs: {}'.format(phase, epoch_loss, epoch_acc, outputs[0:10]))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)
        axes[0].legend(loc='lower right')
        axes[1].legend(loc='lower right')
        plt.savefig('training.svg', format='svg')
        axes[0].cla()
        axes[1].cla()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {}'.format(best_acc))

    return best_model

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def main(args):
    model_ft = models.vgg16(pretrained=True)
    for param in model_ft.parameters():
        param.requires_grad = False
    mod = list(model_ft.classifier.children())
    mod.pop()
    mod.append(torch.nn.Linear(4096, 1))
    new_classifier = torch.nn.Sequential(*mod)
    model_ft.classifier = new_classifier
    for param in model_ft.classifier.parameters():
        param.requires_grad = True
    print(model_ft)
    if use_gpu:
        model_ft = model_ft.cuda()

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.classifier.parameters(), lr=args.learning_rate)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments for the model\'s trainer')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--bs', dest='batch_size', type=int, default=40, help='Batch size')
    parser.add_argument('--ep', dest='epochs', type=int, default=50, help='Number of training epochs')
    args = parser.parse_args()
    main(args)
