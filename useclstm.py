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

# input = Variable(torch.rand(batch_size,seq_len,inp_chans,shape[0],shape[1])).cuda()
input = Variable(torch.rand(batch_size,seq_len,inp_chans,shape[0],shape[1]))

conv_lstm=clstm.CLSTM(shape, inp_chans, filter_size, num_features,nlayers)
conv_lstm.apply(weights_init)
#conv_lstm.cuda()

print('convlstm module:', conv_lstm)


print('params:')
params=conv_lstm.parameters()
for p in params:
   print('param', p.size())
   print('mean',torch.mean(p))


hidden_state=conv_lstm.init_hidden(batch_size)
print('hidden_h shape', len(hidden_state))
print('hidden_h shape', hidden_state[0][0].size())
out=conv_lstm(input,hidden_state)
print('out shape', out[1].size())
print('len hidden', len(out[0]))
print('next hidden', out[0][0][0].size())
print('convlstm dict', conv_lstm.state_dict().keys())
L=torch.sum(out[1])
L.backward()
