with open('resulting_sets', 'r') as f:
    c = f.read().split('\n')[:-1]
c = c[3:]
while '' in c:
    c.remove('')
train_list = []
test_list = []
index = 0
while c[index] != '###Test set###':
    train_list.append(c[index])
    index += 1
index += 1
while c[index] != '###Train distribution###':
    test_list.append(c[index])
    index += 1
with open('trainset_labels', 'r') as f:
    vid = f.read().split('\n')[:-1]
with open('testset_labels', 'r') as f:
    vid += f.read().split('\n')[:-1]
train_set = []
test_set = []
for e in vid:
    for t in train_list:
        if t in e:
            train_set.append(e)
for e in vid:
    for t in test_list:
        if t in e:
            test_set.append(e)
with open('trainset_resplit', 'w') as f:
    for t in train_set:
        f.write('{}\n'.format(t))
with open('testset_resplit', 'w') as f:
    for t in test_set:
        f.write('{}\n'.format(t))
