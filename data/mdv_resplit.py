from simanneal import Annealer
import random
from pyemd import emd
import numpy as np

ACTIONS = {
        "nothing": 0,
        "walking": 0,
        "fighting": 1,
        "picking_up": 1,
        "standing": 0,
        "attacking": 1,
        "talking": 0,
        "stealing": 1,
        "cycling": 1,
        "loitering": 0,
        "running": 1,
        "parking": 0,
        "parked": 0,
        "moving": 0,
        "stopping": 0,
        "falling": 1,
        "repairing": 1,
        "reserving": 0
}

class Action:
    def __init__(self, name):
        self.name = name
        self.label = ACTIONS[self.name]

class Frame:
    def __init__(self, path, actions):
        self.path = path
        self.actions = [Action(a) for a in actions]
        self.label = 0
        for a in self.actions:
            if a.label == 1:
                self.label = 1
                break

class Video:
    def __init__(self, name, frames):
        self.name = name.split('/')[-1]
        self.frames = [Frame(f[0], f[1]) for f in frames]

    def __len__(self):
        return len(self.frames)

    def __str__(self):
        return 'Video {} has {} frames'.format(self.name, len(self.frames))

class Dataset:
    def __init__(self, videos, Videos=None):
        if Videos == None:
            self.videos = [Video(v[0], v[1]) for v in videos]
        else:
            self.videos = Videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        return self.videos[idx]

    def nb_frames(self):
        return sum([len(v) for v in self.videos])

    def get_distribution(self):
        actions = {a:0 for a in sorted(ACTIONS.keys())}
        for v in self.videos:
            for f in v.frames:
                for a in f.actions:
                    actions[a.name] += 1

        return actions

def get_videos(c):
    videos = []
    tmp = []
    name = c[0][0].split('-')[0]
    for b in c:
        if b[0].split('-')[0] == name:
            tmp.append(b)
        else:
            videos.append([name, tmp])
            tmp = [b]
            name = b[0].split('-')[0]
    if len(tmp) > 1:
        videos.append([name, tmp])

    return videos

with open('trainset_labels', 'r') as f:
    c1 = f.read().split('\n')[:-1]
c1 = [[b.split('\t')[0], b.split('\t')[2].split(',')] for b in c1]
with open('testset_labels', 'r') as f:
    c2 = f.read().split('\n')[:-1]
c2 = [[b.split('\t')[0], b.split('\t')[2].split(',')] for b in c2]
train = Dataset(get_videos(c1))
test = Dataset(get_videos(c2))
ds = Dataset(get_videos(c1 + c2))

class Resplit(Annealer):
    def __init__(self, dataset):
        self.dataset = dataset
        self.state = [[0 for v in dataset], [0 for v in dataset]]
        for v in range(len(dataset)):
            r = random.randint(0, 1)
            self.state[r][v] = 1
        self.train = None
        self.test = None
        self.weight = np.ones((len(self.dataset.get_distribution().keys()), len(self.dataset.get_distribution().keys())))

    def toset(self):
        train_videos = [self.dataset[i] for i in range(len(self.state[0])) if self.state[0][i] == 1]
        test_videos = [self.dataset[i] for i in range(len(self.state[1])) if self.state[1][i] == 1]
        self.train = Dataset(None, train_videos)
        self.test = Dataset(None, test_videos)

    def check_independance(self):
        independant = True
        videos = [('Crash_Follow_Day_Half_0_2_1', 'Crash_Follow_Day_Half_0_2_2'),
                  ('Normal_Circle_Day_Half_0_7_2', 'Normal_Circle_Day_Half_0_7_3'),
                  ('Normal_Follow_Day_Half_1_1_1', 'Normal_Follow_Day_Half_1_1_2'),
                  ('StealingPedestrian_Static_Day_Half_0_3_1', 'StealingPedestrian_Static_Day_Half_0_3_2'),
                  ('Suspicious_Static_Day_Half_0_1_1', 'Suspicious_Follow_Day_Half_0_m_1'),
                  ('Suspicious_Static_Day_Half_0_2_1', 'Suspicious_Static_Day_Half_0_2_2'),
                  ('Suspicious_CloseUp_Night_Empty_1_3_1', 'Normal_Static_Night_Empty_1_3_1'),
                  ('Reserving_Static_Day_Half_0_1_2', 'Reserving_Static_Day_Half_0_1_1'),
                  ('Broken_CloseUp_Day_Half_1_1_2', 'Broken_CloseUp_Day_Half_1_1_1'),
                  ('StealingPedestrian_CloseUp_Day_Half_0_2_1', 'StealingPedestrian_CloseUp_Day_Half_0_2_2'),
                  ('Normal_Circle_Day_Half_0_7_1', 'Normal_Circle_Day_Half_0_7_2'),
                  ('Normal_Circle_Day_Half_0_7_1', 'Normal_Circle_Day_Half_0_7_3'),
                  ('Normal_Follow_Day_Half_0_1_1', 'Normal_Static_Day_Half_0_2_1')]
        train_videos = [v.name for v in self.train.videos]
        test_videos = [v.name for v in self.test.videos]
        for v in videos:
            if v[0] in train_videos and v[1] not in train_videos:
                independant = False
            elif v[0] in test_videos and v[1] not in test_videos:
                independant = False

        return independant

    def move(self):
        mode = random.randint(0, 1)
        if mode == 0:
            r1 = random.randint(0, len(self.state[0]) - 1)
            r2 = random.randint(0, len(self.state[0]) - 1)
            self.state[0][r1] = 1 - self.state[0][r1]
            self.state[1][r2] = 1 - self.state[1][r2]
            self.state[0][r2] = 1 - self.state[0][r2]
            self.state[1][r1] = 1 - self.state[1][r1]
        else:
            s = random.randint(0, 1)
            v = random.randint(0, len(self.state[0]) - 1)
            self.state[s][v] = 1 - self.state[s][v]
            self.state[1 - s][v] = 1 - self.state[1 - s][v]

    def energy(self):
        self.toset()
        train_distribution = self.train.get_distribution()
        test_distribution = self.test.get_distribution()
        train = [float(train_distribution[x]) for x in sorted(train_distribution.keys())]
        test = [float(test_distribution[x]) for x in sorted(test_distribution.keys())]
        e = emd(np.array(train), np.array(test), self.weight)
        if not self.check_independance():
            e = float('inf')

        return e

resplit = Resplit(ds)
resplit.Tmax = 100.0  # Max (starting) temperature
resplit.Tmin = 2.5      # Min (ending) temperature
resplit.steps = 1000000  # Number of iterations
resplit.updates = 100
sets, energy = resplit.anneal()
resplit.state = sets
resplit.toset()
with open('resulting_sets', 'w') as f:
    f.write('###EMD: {}###\n'.format(energy))
    f.write('\n###Training set###\n')
    for v in resplit.train.videos:
        f.write('{}\n'.format(v.name))
    f.write('\n###Test set###\n')
    for v in resplit.test.videos:
        f.write('{}\n'.format(v.name))
    f.write('\n###Train distribution###\n')
    train_dist = resplit.train.get_distribution()
    for a in sorted(train_dist.keys()):
        f.write('{}\t{}\n'.format(a, train_dist[a]))
    f.write('\n###Test distribution###\n')
    test_dist = resplit.test.get_distribution()
    for a in sorted(test_dist.keys()):
        f.write('{}\t{}\n'.format(a, test_dist[a]))
    f.write('\nTrainset: {} videos ({} frames)\n'.format(len(resplit.train), sum([len(v) for v in resplit.train.videos])))
    f.write('Testset: {} videos ({} frames)\n'.format(len(resplit.test), sum([len(v) for v in resplit.test.videos])))
print(resplit.train.get_distribution(), resplit.test.get_distribution())
train_distribution = train.get_distribution()
test_distribution = test.get_distribution()
weight = np.ones((len(train_distribution.keys()), len(train_distribution.keys())))
train_d = [float(train_distribution[x]) for x in sorted(train_distribution.keys())]
test_d = [float(test_distribution[x]) for x in sorted(test_distribution.keys())]
e = emd(np.array(train_d), np.array(test_d), weight)
print(e)
