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
import matplotlib.pyplot as plt
from scipy import misc

with open('data/trainset_labels', 'r') as f:
    trainset = f.read().split('\n')[:-1]
trainset = [(c.split('\t')[0], int(c.split('\t')[1])) for c in trainset]
with open('data/valset_labels', 'r') as f:
    valset = f.read().split('\n')[:-1]
valset = [(c.split('\t')[0], int(c.split('\t')[1])) for c in valset]
dsets = {'train': trainset, 'val': valset}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}

use_gpu = torch.cuda.is_available()

def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=25):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            step = 0
            max_step = int(dset_sizes[phase] / 40)

            # Iterate over data.
            while step < max_step:
                # get the inputs
                idx_start = step * 40
                idx_end = idx_start + 40
                inputs = np.array([misc.imread(os.path.join('data', dsets[phase][i][0] + '.png')) for i in range(idx_start, idx_end)])
                inputs = np.transpose(inputs, (0, 3, 1, 2))
                labels = np.array([dsets[phase][i][1] for i in range(idx_start, idx_end)])
                print(labels.shape)
                labels = np.reshape(labels, (40, 1))
                print(labels.shape)
                #convert to tensor
                inputs, labels = torch.from_numpy(inputs).float(), torch.from_numpy(labels).long()
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
                loss = criterion(torch.sigmoid(outputs).long(), labels)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                # next step
                step += 1
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return best_model

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

model_ft = models.vgg16(pretrained=True)
mod = list(model_ft.classifier.children())
mod.pop()
mod.append(torch.nn.Linear(4096, 1))
new_classifier = torch.nn.Sequential(*mod)
model_ft.classifier = new_classifier
print(model_ft)
if use_gpu:
    model_ft = model_ft.cuda()

#criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
