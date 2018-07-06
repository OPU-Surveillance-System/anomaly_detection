from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from scipy import misc

from torch.autograd import Variable

def minidronevideo(frames):
    video = frames[0][0].split('/')[-1].split('-')[0]
    groups = []
    tmp = [frames[0]]
    for c in frames[1:]:
        if c[0].split('/')[-1].split('-')[0] == video:
            tmp.append(c)
        else:
            groups.append(tmp)
            video = c[0].split('/')[-1].split('-')[0]
            tmp = [c]
    groups.append(tmp)

    return groups

def umn(frames):
    pattern = frames[0][0].split('_')[1]
    video = frames[0][0].split('_')[2]
    groups = []
    tmp = [frames[0]]
    for c in frames[1:]:
        if c[0].split('_')[1] == pattern and c[0].split('_')[2] == video:
            tmp.append(c)
        else:
            groups.append(tmp)
            pattern = c[0].split('_')[1]
            video = c[0].split('_')[2]
            tmp = [c]
    groups.append(tmp)

    return groups

class MiniDroneVideoDataset(Dataset):
    """
    """

    def __init__(self, dataset, summary, root_dir, sequence_length, stride, transform=None):
        """
        """

        self.dataset = dataset
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.stride = stride
        self.root_dir = root_dir
        with open(summary, 'r') as f:
            content = f.read().split('\n')[:-1]
        self.content = [c.split('\t') for c in content]
        #Group frames from the same video
        if self.dataset == 'minidrone':
            groups = minidronevideo(self.content)
        elif self.dataset == 'umn':
            groups = umn(self.content)
        #Build sequences from the same video of length sequence_length
        self.frames = [(['data/' + f[0] + '.png' for f in g[i*self.stride:(i*self.stride)+self.sequence_length]], [int(f[1]) for f in g[i*self.stride:(i*self.stride)+self.sequence_length]])
                       for g in groups for i in range(int(len(g) / self.stride))]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        images = np.array([io.imread(f)*1/255 for f in self.frames[idx][0]], dtype=np.float)
        labels = np.array([f for f in self.frames[idx][1]], dtype=np.float)
        labels = labels.reshape(len(labels), 1)
        names = [os.path.basename(f) for f in self.frames[idx][0]]
        sample = {'images': images, 'labels':labels, 'names':names}
        if self.transform:
            sample = self.transform(sample)
        normalized = []
        for i in sample['images']:
            x = i
            x -= x.mean()
            x /= x.std()
            normalized.append(x)
        sample['images'] = np.array(normalized)
        sample['images'] = sample['images'].transpose((0, 3, 1, 2))

        return sample

class NegativeDataset(Dataset):
    """
    """

    def __init__(self, summary, root_dir):
        """
        """

        self.root_dir = root_dir
        with open(summary, 'r') as f:
            content = f.read().split('\n')[:-1]
        self.content = [c.split('\t') for c in content]
        self.normal = ['data/{}.png'.format(c[0]) for c in self.content if c[1] == '0']
        self.abnormal = ['data/{}.png'.format(c[0]) for c in self.content if c[1] == '1']
        self.active = 0

    def __len__(self):
        if self.active == 0:
            return len(self.normal)
        else:
            return len(self.abnormal)

    def __getitem__(self, idx):
        if self.active == 0:
            dataset = self.normal
            label = 0
        else:
            dataset = self.abnormal
            label = 1
        x = misc.imread(dataset[idx])#.reshape(224, 224, 1)
        x = (x - x.min()) / (x.max() - x.min())
        name = dataset[idx]

        x -= x.mean()
        x /= x.std()

        x = x.transpose((2, 0, 1))

        sample = {'image': x, 'label':label, 'name': name}

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        images = sample['images']
        h, w = images[0].shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        rescaled = []
        for i in images:
            rescaled.append(transform.resize(i, (new_h, new_w), mode='constant'))
        rescaled = np.array(rescaled)

        return {'images': rescaled, 'labels':sample['labels'], 'names':sample['names']}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        images = sample['images']
        if 0.5 > np.random.uniform(0.0, 1.0):
            h, w = images[0].shape[:2]
            new_h, new_w = self.output_size
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            croped = []
            for i in images:
                croped.append(i[top: top + new_h, left: left + new_w])
            croped = np.array(croped)
            rescale = Rescale((224, 224))
            rescaled = rescale({'images': croped, 'labels': sample['labels'], 'names':sample['names']})
            rescaled = rescaled['images']
        else:
            rescaled = np.array(images)

        return {'images': rescaled, 'labels': sample['labels'], 'names':sample['names']}

class RandomFlip(object):
    """Flip randomly the image in a sample.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        images = sample['images']
        if 0.5 > np.random.uniform(0.0, 1.0):
            fliped = []
            for i in images:
                fliped.append(np.fliplr(i))
            fliped = np.array(fliped)
        else:
            fliped = np.array(images)

        return {'images': fliped, 'labels': sample['labels'], 'names':sample['names']}

class Dropout(object):
    """Flip randomly the image in a sample.
    """

    def __init__(self, amount):
        self.amount = amount

    def __call__(self, sample):
        images = sample['images']
        if 0.5 > np.random.uniform(0.0, 1.0):
            droped = []
            num_droped = np.ceil(np.random.uniform(0.0, self.amount) * 224*224)
            coords = [np.random.randint(0, i - 1 , int(num_droped)) for i in images[0].shape]
            for i in images:
                i[coords[:-1]] = (0, 0, 0)
                droped.append(i)
            droped = np.array(droped)
        else:
            droped = np.array(images)

        return {'images': droped, 'labels': sample['labels'], 'names':sample['names']}

class Normalization(object):
    """Flip randomly the image in a sample.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        images = sample['images']
        normalized = []
        for i in images:
            i -= self.mean
            i /= self.std
            normalized.append(i)
        normalized = np.array(normalized)

        return {'images': normalized, 'labels': sample['labels'], 'names':sample['names']}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return {'images': torch.from_numpy(sample['images']), 'labels': torch.from_numpy(sample['labels']), 'names':sample['names']}

# ds = MiniDroneVideoDataset('umn', 'data/umn_trainset_labels',
#                                 'data',
#                                     20, 20,
#                                     transform=transforms.Compose([RandomCrop((160, 160)), RandomFlip(), Dropout(0.2)]))
# fig = plt.figure()
# sample = ds[50]
# #print(max(set(sample['images'][0].flatten())))
# for i in range(len(sample['images'])):
#     plt.imsave('{}.png'.format(sample['names'][i]), sample['images'][i].transpose(1, 2, 0))
