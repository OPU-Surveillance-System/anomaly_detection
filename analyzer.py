import argparse
import os
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

import plot

def accuracy(threshold, scores, labels):
    scores = np.array([float(i > threshold) for i in scores])
    accuracy = accuracy_score(labels, scores)

    return accuracy

def main(args):
    with open(os.path.join(args.directory, 'res_summary'), 'r') as f:
        content = f.read().split('\n')[:-1]
    res = np.array([[float(elt) for elt in c.split('\t')[1:]] for c in content])
    names = sorted([c.split('\t')[0] for c in content])
    # with open('test', 'w') as f:
    #     for n in sorted(names):
    #         f.write('{}\n'.format(n))
    # with open(os.path.join(args.directory, 'thresholds'), 'r') as f:
    #     content = f.read().split('\n')[:-1]
    # thresholds = [float(c) for c in content]
    # accuracies = {}
    # for thr in tqdm(thresholds):
    #     accuracies[thr] = accuracy(thr, res[:, 0], res[:, 1])
    # plot.plot_accuracy_per_threshold(accuracies, args.directory, 'accuracies')
    # best = 0
    # best_thr = 0
    # for thr in thresholds:
    #     if accuracies[thr] > best:
    #         best = accuracies[thr]
    #         best_thr = thr
    # print(names)
    # print('Best threshold: {} ({})'.format(best_thr, accuracies[best_thr]))
    sequences = []
    tmp = [names[0]]
    current = names[0].split('-')[0]
    for n in range(1, len(names) - 1):
        if names[n].split('-')[0] == current:
            tmp.append(names[n])
        else:
            current = names[n].split('-')[0]
            sequences.append(tmp)
            tmp = [names[n]]
    sequences.append(tmp)
    with open(os.path.join(args.directory, 'res_summary'), 'r') as f:
        content = f.read().split('\n')[:-1]
    res = [c.split('\t') for c in content]
    scores = {r[0]:float(r[1]) for r in res}
    labels = {r[0]:int(float(r[2])) for r in res}
    for i in range(len(sequences)):
        sequences[i] = sorted(sequences[i], key=lambda x: int(x.split('-')[-1][:-4]))
        sco = [scores[s] for s in sequences[i]]
        la = [labels[s] for s in sequences[i]]
        x = list(range(len(sequences[i])))
        thr = [0.4326460063457489 for s in range(len(sequences[i]))]
        plt.plot(x, sco)
        plt.plot(x, la, color='green')
        plt.plot(x, thr, '--', color='red')
        plt.title('{}'.format(sequences[i]).split('-')[0][2:])
        plt.show()
    # tp = []
    # tn = []
    # fp = []
    # fn = []
    # for i in range(len(content)):
    #     if res[i][0] >= best_thr and res[i][1] == 1:
    #         tp.append([names[i], res[i][0]])
    #     elif res[i][0] < best_thr and res[i][1] == 0:
    #         tn.append([names[i], res[i][0]])
    #     elif res[i][0] >= best_thr and res[i][1] == 0:
    #         fp.append([names[i], res[i][0]])
    #     else:
    #         fn.append([names[i], res[i][0]])
    # tp.sort(key=lambda x: x[0])
    # tn.sort(key=lambda x: x[0])
    # fp.sort(key=lambda x: x[0])
    # fn.sort(key=lambda x: x[0])
    # print('{} true positives, {} true negatives, {} false positives, {} false negatives'.format(len(tp), len(tn), len(fp), len(fn)))
    # with open(os.path.join(args.directory, 'tp'), 'w') as f:
    #     for elt in tp:
    #         f.write('{}\t{}\n'.format(elt[0], elt[1]))
    # with open(os.path.join(args.directory, 'tn'), 'w') as f:
    #     for elt in tn:
    #         f.write('{}\t{}\n'.format(elt[0], elt[1]))
    # with open(os.path.join(args.directory, 'fp'), 'w') as f:
    #     for elt in fp:
    #         f.write('{}\t{}\n'.format(elt[0], elt[1]))
    # with open(os.path.join(args.directory, 'fn'), 'w') as f:
    #     for elt in fn:
    #         f.write('{}\t{}\n'.format(elt[0], elt[1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments for the model\'s trainer')
    parser.add_argument('--dir', dest='directory', type=str, default='experiment', help='Path to a directory for saving results')
    args = parser.parse_args()
    main(args)
