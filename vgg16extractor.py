import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

class VGG16Extractor():
    """
    """

    def __init__(self, margs):
        """
        --bn: Batch Norm (bool)
        --do: Dropout prob (float)
        --trl: Trainable layers (int)
        --wl: Weighted loss (bool)
        --lr: Learning rate (float)
        --thr: Alert threshold (float)
        """

        if 'bn' not in margs.keys():
            margs['bn'] = False
        if 'do' not in margs.keys():
            margs['do'] = 0.5
        if 'trl' not in margs.keys():
            margs['trl'] = 1
        if 'wl' not in margs.keys():
            margs['wl'] = False
        if 'lr' not in margs.keys():
            margs['lr'] = 0.000001
        if 'thr' not in margs.keys():
            margs['thr'] = 0.5

        self.margs = margs
        #Load pretrained VGG16
        model = models.vgg16(pretrained=True)
        #Freeze convolutional layers
        for param in model.parameters():
            param.requires_grad = False
        #Extract classifier part
        mod = list(model.classifier.children())
        #Use dropout
        mod[2] = torch.nn.Dropout(float(self.margs['do']))
        mod[5] = torch.nn.Dropout(float(self.margs['do']))
        #Add batch norm if specified
        if bool(self.margs['bn']):
            mod.insert(1, torch.nn.BatchNorm1d(4096))
            mod.insert(5, torch.nn.BatchNorm1d(4096))
        #Change the final layer
        mod.pop()
        mod.append(torch.nn.Linear(4096, 1))
        new_classifier = torch.nn.Sequential(*mod)
        #Replace the classifier part
        model.classifier = new_classifier
        #Set specified parameters trainable
        parameters = list(model.classifier.parameters())
        if bool(self.margs['bn']):
            tmp = [parameters[9], parameters[8]]
            parameters = [parameters[7 - p] for p in range((int(self.margs['trl']) - 1) * 4)]
            parameters = list(reversed(tmp + parameters))
        else:
            parameters = [parameters[5 - p] for p in range(int(self.margs['trl']) * 2)]
        for param in parameters:
            param.requires_grad = True
        model = model.cuda()
        self.model = model
        #Cross entropy function
        if bool(self.margs['wl']):
            weight = [0.343723, 0.656277]
        else:
            weight = [1, 1]
        self.weght = weight
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(parameters, lr=float(self.margs['lr']))

    def predict(self, probs):
        """
        """

        return (torch.sigmoid(probs.data) > float(self.margs['thr'])).long()
