import torch
from torch.autograd import Variable

import convlstm as clstm

num_features=10
filter_size=5
batch_size=10
shape=(25,25)#H,W
inp_chans=3
nlayers=2
seq_len=4

convlstm=clstm.Conv2dLSTMCell(in_channels=2, out_channels=1, kernel_size=(3, 3), stride=1, padding=0)
convlstm.cuda()
x = Variable(torch.ones(1, 2)).cuda()
h = Variable(torch.ones(1, 1)).cuda()
c = Variable(torch.ones(1, 1)).cuda()
t = Variable(torch.zeros(1, 1)).cuda()
x.data.resize_(1, 2, 22, 22).random_()
h.data.resize_(1, 1, 20, 20).random_()
c.data.resize_(1, 1, 20, 20).random_()
t.data.resize_(1, 1, 20, 20).random_()

y, hx = convlstm(x, (h, c))
loss = (y - t).mean()
loss.backward()
print('convlstm module:', conv_lstm)
