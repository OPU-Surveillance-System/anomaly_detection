import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

class VGG16LSTM(nn.Module):
    """
    """

    def __init__(self, margs):
        """
        --bn: Batch Norm (bool)
        --do: Dropout prob (float)
        --wl: Weighted loss (bool)
        --lr: Learning rate (float)
        --thr: Detection threshold (float)
        --hs: LSTM Hidden state's size
        --rl: LSTM's number of layers
        """

        if 'bn' not in margs.keys(): #Batch norm
            margs['bn'] = False
        if 'do' not in margs.keys(): #Dropout
            margs['do'] = 0.0
        if 'wl' not in margs.keys(): #weighted loss
            margs['wl'] = False
        if 'thr' not in margs.keys(): #Detection threshold
            margs['thr'] = 0.5
        if 'hd' not in margs.keys(): #LSTM Hidden state's dimension
            margs['hd'] = 100
        if 'rl' not in margs.keys(): #LSTM's number of layers
            margs['rl'] = 1

        super(VGG16LSTM, self).__init__()
        self.margs = margs
        #VGG part
        self.vgg = models.vgg16(pretrained=True) #Load pretrained VGG16
        mod = list(self.vgg.classifier.children()) #Remove the final layer
        mod.pop()
        new_classifier = torch.nn.Sequential(*mod)
        self.vgg.classifier = new_classifier
        for param in self.vgg.parameters(): #Freeze convolutional layers
            param.requires_grad = False
        #LSTM part
        self.rnn = nn.LSTM(input_size=4096,
                           hidden_size=self.margs['hd'],
                           num_layers=self.margs['rl'],
                           dropout=self.margs['do'])
        self.hidden = self.init_hidden()
        self.trainable_parameters = list(self.rnn.parameters())
        self.out_layer = nn.Linear(self.margs['hd'], 1)
        self.trainable_parameters += list(self.out_layer.parameters())

    def init_hidden(self):
        """
        """

        #Dimension semantic: Number of recurent layers, Batch size, Hidden size
        init = (Variable(torch.zeros(self.margs['rl'], 1, self.margs['hd'])),
                Variable(torch.zeros(self.margs['rl'], 1, self.margs['hd'])))

        return init

    def forward(self, frames):
        """
        input dimensions: [Sequence, Height, Width, Channels]
        """

        frames = np.transpose(frames, (0, 3, 1, 2))
        embeds = self.vgg(frames)
        lstm_out, self.hidden = self.lstm(embeds.view(len(frames), 1, -1), self.hidden)
        logits = self.out_layer(lstm_out.view(len(frames), -1))

        return logits

    def predict(self, probs):
        """
        """

        return (torch.sigmoid(probs.data) > float(self.margs['thr'])).long()
