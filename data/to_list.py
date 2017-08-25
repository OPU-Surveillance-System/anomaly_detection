"""
Save data paths and labels into files.
"""

import argparse
import os
import numpy as np
from tqdm import tqdm

def main(args):
    with open(args.inf, 'r') as f:
        content = f.read().split('\n')[:-1]
    summary = [c.split('\t') for c in content]
    print('Group frames from the same video together')
    video = summary[0][0].split('-')[0]
    group = [summary[0]]
    tmp = []
    for s in summary[1:]:
        if s[0].split('-')[0] == video and s not in group:
            group.append(s)
        else:
            tmp.append(group)
            video = s[0].split('-')[0]
            group = [s]
    tmp.append(group)
    summary = tmp
    verif = True
    with open(args.outf, 'w') as f:
        for s in tqdm(summary):
            for w in range(len(s) - (args.wl - 1)):
                ex = s[w:w + args.wl]
                line = '\t'.join(['data/' + e[0] + '.png' for e in ex])
                label = sum([int(e[1]) for e in ex])
                if label > 1:
                    label = 1
                line = '\t'.join([line, str(label)])
                f.write(line + '\n')
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments for the dataset spliter')
    parser.add_argument('--inf', dest='inf', default='', help='Input label list file')
    parser.add_argument('--outf', dest='outf', default='', help='Output file')
    parser.add_argument('--wl', dest='wl', type=int, default=1, help='sliding windows lenght')
    args = parser.parse_args()
    main(args)
