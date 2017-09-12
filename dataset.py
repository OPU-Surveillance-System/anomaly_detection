from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class MiniDroneVideoDataset(Dataset):
    """
    """

    def __init__(self, summary, root_dir, sequence_length, transform=None):
        """
        """

        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        with open(summary, 'r') as f:
            content = f.read().split('\n')[:-1]
        self.content = [c.split('\t') for c in content]
        #Group frames from the same video
        video = self.content[0][0].split('/')[-1].split('-')[0]
        groups = []
        tmp = [self.content[0]]
        for c in self.content[1:]:
            if c[0].split('/')[-1].split('-')[0] == video:
                tmp.append(c)
            else:
                groups.append(tmp)
                video = c[0].split('/')[-1].split('-')[0]
                tmp = [c]
        groups.append(tmp)
        #Build sequences from the same video of length sequence_length
        self.frames = [([os.path.join(self.root_dir, f + '.png') for f in g[i][0:self.sequence_length]], [int(f) for f in g[i][self.sequence_length:]])
                       for g in groups for i in range(len(g) - (self.sequence_length - 1))]
        print(self.frames[0])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        images = np.array([io.imread(f) for f in self.frames[idx][0]])
        labels = np.array([f for f in self.frames[idx][1]], dtype=np.float)
        labels = labels.reshape(len(labels), 1)
        sample = {'images': images, 'labels':labels}
        if self.transform:
            sample = self.transform(sample)
        sample['images'] = np.transpose(sample['images'], (0, 3, 1, 2))

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
            rescaled.append(transform.resize(i, (new_h, new_w)))
        rescaled = np.array(rescaled)

        return {'images': rescaled, 'labels':sample['labels']}


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
            rescaled = rescale({'images': croped, 'labels': sample['labels']})
            rescaled = rescaled['images']
        else:
            rescaled = np.array(images)

        return {'images': rescaled, 'labels': sample['labels']}

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

        return {'images': fliped, 'labels': sample['labels']}

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

        return {'images': droped, 'labels': sample['labels']}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['images']
        image = image.transpose((2, 0, 1))
        return {'images': torch.from_numpy(image),
                'labels': torch.from_numpy(sample['labels'])}

# ds = MiniDroneVideoDataset('data/trainset_labels',
#                                 'data',
#                                     10,
#                                     transform=transforms.Compose([
#                                            RandomCrop((160, 160)),
#                                            RandomFlip(),
#                                            Dropout(0.2)
#                                        ]))
# ds[0]
# scale = Rescale((512, 600))
# crop = RandomCrop(128)
# flip = RandomFlip()
# composed = transforms.Compose([RandomCrop((160, 160)), RandomFlip(), Dropout(0.2)])

# Apply each of the above transforms on sample.
# fig = plt.figure()
# sample = ds[120]
# for i, tsfrm in enumerate([composed]):
#     transformed_sample = tsfrm(sample)
#     for j in range(len(transformed_sample['images'])):
#         plt.imsave('{}.png'.format(j), transformed_sample['images'][j])
