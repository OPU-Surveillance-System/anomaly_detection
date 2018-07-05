import argparse
import os
import torch
import copy
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from sklearn import metrics
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import dataset
from models import autoencoder

def plot_reconstruction_images(inputs, pred, name):
    """
    Plot example of reconstruction images
    Args:
        inputs (numpy.array): True images
        pred (numpy.array): Reconstructed images
        name (str): name to save the figure (if None: show the figure)
    """

    plt.clf()
    nb_plots = min(inputs.shape[0], 4)
    #inputs
    for i in range(nb_plots):
        ax = plt.subplot2grid((2, nb_plots), (0, i), rowspan=1, colspan=1)
        ax.imshow(inputs[i])
        ax.axis('off')
    #pred
    for i in range(nb_plots):
        ax = plt.subplot2grid((2, nb_plots), (1, i), rowspan=1, colspan=1)
        ax.imshow(pred[i])
        ax.axis('off')

    if name != None:
        plt.savefig(name, format='svg', bbox_inches='tight')
    else:
        plt.show()

parser = argparse.ArgumentParser(description='')
#Training arguments
parser.add_argument('--bs', dest='batch_size', type=int, default=16, help='Mini batch size')
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--ep', dest='epoch', type=int, default=100, help='Number of training epochs')
parser.add_argument('--dir', dest='directory', type=str, default='negative_learning', help='Directory to store results')
#Model arguments
parser.add_argument('-f', dest='f', type=int, default=8, help='Number of hidden features')
parser.add_argument('-b', dest='b', type=int, default=2, help='Number of blocks')
parser.add_argument('-l', dest='l', type=int, default=2, help='Number of layers per block')
parser.add_argument('-z', dest='z', type=int, default=2, help='Latent size')

args = parser.parse_args()

#Create directories if it don't exists
if not os.path.exists(args.directory):
    os.makedirs(args.directory)
if not os.path.exists(os.path.join(args.directory, 'serial')):
    os.makedirs(os.path.join(args.directory, 'serial'))
if not os.path.exists(os.path.join(args.directory, 'reconstruction_train')):
    os.makedirs(os.path.join(args.directory, 'reconstruction_train'))
if not os.path.exists(os.path.join(args.directory, 'reconstruction_train', 'normal')):
    os.makedirs(os.path.join(args.directory, 'reconstruction_train', 'normal'))
if not os.path.exists(os.path.join(args.directory, 'reconstruction_train', 'abnormal')):
    os.makedirs(os.path.join(args.directory, 'reconstruction_train', 'abnormal'))
if not os.path.exists(os.path.join(args.directory, 'reconstruction_test')):
    os.makedirs(os.path.join(args.directory, 'reconstruction_test'))
if not os.path.exists(os.path.join(args.directory, 'reconstruction_test', 'normal')):
    os.makedirs(os.path.join(args.directory, 'reconstruction_test', 'normal'))
if not os.path.exists(os.path.join(args.directory, 'reconstruction_test', 'abnormal')):
    os.makedirs(os.path.join(args.directory, 'reconstruction_test', 'abnormal'))
if not os.path.exists(os.path.join(args.directory, 'logs')):
    os.makedirs(os.path.join(args.directory, 'logs'))

trainset = dataset.NegativeDataset('data/trainset_labels', '/home/scom/opu_surveillance_system/anomaly_detection')
testset = dataset.NegativeDataset('data/testset_labels', '/home/scom/opu_surveillance_system/anomaly_detection')

#Write arguments in a file
d = vars(args)
with open(os.path.join(args.directory, 'hyper-parameters'), 'w') as f:
    for k in d.keys():
        f.write('{}:{}\n'.format(k, d[k]))

#Variables
if args.z == 0:
    z = None
else:
    z = args.z
ae = autoencoder.Autoencoder(args.f, args.l, args.b, z)
ae = ae.cuda()
print(ae)

optimizer = torch.optim.Adam(ae.parameters(), args.learning_rate)

dist = torch.nn.PairwiseDistance(p=2, eps=1e-06, keepdim=True)

phase = ('train', 'test')
sets = {'train':trainset, 'test':testset}
modes = {0:'Positive', 1:'Negative'}
classes = {0:'normal', 1:'abnormal'}

writer = SummaryWriter(os.path.join(args.directory, 'logs'))

#Warm up training (1 epoch of Positive learning)
print('Warm up training')
trainset.active = 0
dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
for i_batch, sample in enumerate(tqdm(dataloader)):
    image = sample['image'].float().cuda()

    reconstruction = ae(image)

    loss = torch.nn.functional.mse_loss(reconstruction, image)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#Alternate between Negative and Positive training
for e in range(args.epoch):

    errors = []
    groundtruth = []

    for p in phase:
        running_loss = 0
        ae.train(p == 'train')

        reconstruction_errors = []
        labels = []

        total_loss = 0

        for m in reversed(sorted(modes.keys())):
            running_loss = 0

            sets[p].active = m

            dataloader = DataLoader(sets[p], batch_size=args.batch_size, shuffle=True, num_workers=4)
            for i_batch, sample in enumerate(tqdm(dataloader)):
                image = sample['image'].float().cuda()
                reconstruction = ae(image)

                loss = torch.nn.functional.mse_loss(reconstruction, image)

                # if m == 1:
                #     loss *= -1 #Negative learning

                if p == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                #Get reconstruction error
                reconstruction_errors += dist(image.view(image.size(0), -1), reconstruction.view(reconstruction.size(0), -1)).view(image.size(0)).detach().cpu().numpy().tolist()
                labels += sample['label'].numpy().tolist()

                #Plot reconstructed images
                if i_batch == 0:
                    image = image.permute(0, 2, 3, 1).view(image.size(0), 224, 224)
                    reconstruction = reconstruction.permute(0, 2, 3, 1).view(reconstruction.size(0), 224, 224)
                    plot_reconstruction_images(image.cpu().numpy(), reconstruction.detach().cpu().numpy(), os.path.join(args.directory, 'reconstruction_{}'.format(p), classes[m], 'epoch_{}.svg'.format(e)))

                running_loss += loss.item()

            running_loss /= (i_batch + 1)

            print('Epoch {}, Phase: {}, Mode: {}, loss: {}'.format(e, p, modes[m], running_loss))
            writer.add_scalar('{}/{}_loss'.format(p, modes[m]), running_loss, e)

            total_loss += loss.item()

        writer.add_scalar('{}/total_loss'.format(p), total_loss, e)

        #Compute AUC
        fpr, tpr, thresholds = metrics.roc_curve(labels, reconstruction_errors)
        auc = metrics.auc(fpr, tpr)
        writer.add_scalar('{}/auc'.format(p), auc, e)

writer.export_scalars_to_json(os.path.join(args.directory, 'logs', 'scalars.json'))
writer.close()
