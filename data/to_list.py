"""
Save data paths and labels into files.
"""

import argparse
import os
import numpy as np
from tqdm import tqdm

def main(args):
    subsets = args.subset.split(',')
    sets = [[p + '/' + frame for p, n, f in os.walk(args.dataset) for frame in f if s in p and args.format in frame] for s in subsets]
    for s in tqdm(range(len(sets))):
        setname = subsets[s]
        frames = [f for f in sets[s]]
        with open('%sset_labels'%(setname), 'r') as f:
            labels = f.read()
        labels = [l.split('\t')[1] for l in labels.split('\n')[:-1]]
        dataset = ['%s\t%s\n'%(frames[i], labels[i]) for i in range(len(frames))]
        with open('%sset_list'%(setname), 'w') as f:
            for elt in dataset:
                f.write(elt)
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments for the dataset spliter')
    parser.add_argument('--dataset', dest='dataset', default='MiniDrone_frames', help='Path to the dataset to split')
    parser.add_argument('--format', dest='format', default='.png', help='Frames format')
    parser.add_argument('--subset', dest='subset', default='train,val,test', help='Subsets for the dataset to be divided in')
    args = parser.parse_args()
    main(args)
