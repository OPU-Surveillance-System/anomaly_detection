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
import imgaug as ia
from imgaug import augmenters as iaa

use_gpu = torch.cuda.is_available()

def train_model(model, criterion, optimizer, lr_scheduler):
    with open(args.trainset, 'r') as f:
        trainset = f.read().split('\n')[:-1]
    trainset = [(c.split('\t')[0], int(c.split('\t')[1])) for c in trainset]
    with open(args.valset, 'r') as f:
        valset = f.read().split('\n')[:-1]
    valset = [(c.split('\t')[0], int(c.split('\t')[1])) for c in valset]
    dsets = {'train': trainset, 'val': valset}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    since = time.time()

    best_model = model
    best_acc = 0.0
    summaries = {'train':{'loss':[], 'accuracy':[]}, 'val':{'loss':[], 'accuracy':[]}}

    fig, axes = plt.subplots(nrows=1, ncols=2)
    color = {'train':'blue', 'val':'red'}
    m = nn.Sigmoid()

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
            #Data augmenters
            if phase == 'train' and args.augdata:
                seq = iaa.Sequential(iaa.SomeOf((0, 2), [
                    iaa.Fliplr(0.5),
                    #iaa.Invert(0.5),
                    iaa.OneOf([
                        iaa.Add((-5, 25)),
                        iaa.Multiply((0.25, 1.5))
                    ]),
                    iaa.OneOf([
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 1.0)),
                            iaa.AverageBlur(k=(1, 5)),
                            iaa.MedianBlur(k=(1, 5)),
                            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), per_channel=0.5),
                            iaa.Dropout((0.01, 0.3), per_channel=0.5),
                            iaa.CoarseDropout((0.03, 0.15), size_percent=(0.03, 0.05), per_channel=0.2)
                        ]),
                        iaa.OneOf([
                            iaa.Sharpen(alpha=(0, 1.0), lightness=(1.0, 1.75)),
                            iaa.Emboss(alpha=(0, 1.0), strength=(0.2, 0.75)),
                            iaa.EdgeDetect(alpha=(0.1, 0.3)),
                            iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
                            iaa.Grayscale(alpha=(0.0, 1.0)),
                        ])
                    ]),
                    iaa.OneOf([
                        iaa.PiecewiseAffine(scale=(0.01, 0.03)),
                        iaa.Affine(scale=(1.0, 1.3),
                                   translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                                   rotate=(-15, 15),
                                   shear=(-15, 15),
                                   order=[0, 1],
                                   cval=(0, 255),
                                   mode=ia.ALL),
                        iaa.ElasticTransformation(alpha=(0.01, 2.0), sigma=0.25)
                    ]),
                ]), random_order=True)

            #Local variables
            running_loss = 0.0
            running_corrects = 0
            step = 0
            max_step = int(dset_sizes[phase] / args.batch_size)

            # Iterate over data.
            while step < max_step:
                # get the inputs
                idx_start = step * args.batch_size
                idx_end = idx_start + args.batch_size
                shuffle(dsets[phase])
                inputs = np.array([misc.imread(os.path.join('data', dsets[phase][i][0] + '.png')) for i in range(idx_start, idx_end)])
                if phase == 'train' and args.augdata:
                    inputs = seq.augment_images(inputs)
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
                preds = torch.sigmoid(outputs.data)
                loss = criterion(outputs, labels)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds.long() == labels.data.long())
                # next step
                step += 1
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]
            summaries[phase]['loss'].append(epoch_loss)
            summaries[phase]['accuracy'].append(epoch_acc)
            lineloss = axes[0].plot(list(range(epoch + 1)), summaries[phase]['loss'], color=color[phase], label='%s loss'%(phase))
            lineacc = axes[1].plot(list(range(epoch + 1)), summaries[phase]['accuracy'], color=color[phase], label='%s accuracy'%(phase))

            print('{} Loss: {} Acc: {}, some outputs: {}, their corresponding sig: {} and their groundtruths: {}'.format(phase, epoch_loss, epoch_acc, outputs[0:10], preds[0:10], labels[0:10]))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)
        axes[0].legend(loc='lower right')
        axes[1].legend(loc='lower right')
        plt.savefig(os.path.join(args.directory, 'training.svg'), format='svg')
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
    #Create experiment directory
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    #Load pretrained VGG16
    model_ft = models.vgg16(pretrained=True)
    #Freeze convolutional layers
    for param in model_ft.parameters():
        param.requires_grad = False
    #Extract classifier part
    mod = list(model_ft.classifier.children())
    #Use dropout
    mod[2] = torch.nn.Dropout(args.drop_prob)
    mod[5] = torch.nn.Dropout(args.drop_prob)
    #Add batch norm if specified
    if args.batch_norm:
        mod.insert(1, torch.nn.BatchNorm1d(4096))
        mod.insert(5, torch.nn.BatchNorm1d(4096))
    #Change the final layer
    mod.pop()
    mod.append(torch.nn.Linear(4096, 1))
    new_classifier = torch.nn.Sequential(*mod)
    #Replace the classifier part
    model_ft.classifier = new_classifier
    #Set specified parameters trainable
    parameters = list(model_ft.classifier.parameters())
    if args.batch_norm:
        tmp = [parameters[9], parameters[8]]
        parameters = [parameters[7 - p] for p in range((args.trainable_parameters - 1) * 4)]
        parameters = list(reversed(tmp + parameters))
    else:
        parameters = [parameters[5 - p] for p in range(args.trainable_parameters * 2)]
    for param in parameters:
        param.requires_grad = True
    if use_gpu:
        model_ft = model_ft.cuda()
    #Cross entropy function
    if args.weighted_loss:
        weight = [0.343723, 0.656277]
    else:
        weight = [1, 1]
    criterion = nn.BCEWithLogitsLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(parameters, lr=args.learning_rate)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler)
    torch.save(model_ft, os.path.join(args.directory, 'modelsave'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments for the model\'s trainer')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--bs', dest='batch_size', type=int, default=40, help='Batch size')
    parser.add_argument('--ep', dest='epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--tr', dest='trainset', type=str, default='data/trainset_labels', help='Path to the trainset summary')
    parser.add_argument('--val', dest='valset', type=str, default='data/valset_labels', help='Path to the valset summary')
    parser.add_argument('--dir', dest='directory', type=str, default='experiment', help='Path to a directory for saving results')
    parser.add_argument('--doprob', dest='drop_prob', type=float, default='0.5', help='Dropout keep probability')
    parser.add_argument('--trp', dest='trainable_parameters', type=int, default=3, help='Trainable parameters (range in [1, 3] - FC3 to FC1)')
    parser.add_argument('--bn', dest='batch_norm', type=bool, default=False, help='Whether to use batch normalization or not')
    parser.add_argument('--da', dest='augdata', type=bool, default=False, help='Whether to activate data augmentation pipeline or not during training')
    parser.add_argument('--wl', dest='weighted_loss', type=bool, default=False, help='Whether to weight the loss or not')
    args = parser.parse_args()
    main(args)
