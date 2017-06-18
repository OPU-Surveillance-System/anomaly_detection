"""
Save data and labels into tfrecords files.
"""

import numpy as np
import tensorflow as tf
from scipy import misc
from tqdm import tqdm

def main(args):
    subsets = args.subset.split(',')
    sets = [[p + '/' + frame for p, n, f in os.walk(args.dataset) for frame in f if s in p and args.format in frame] for s in subsets]
    for s in tqdm(range(len(sets))):
        setname = subsets[s]
        frames = [misc.imread(f) for f in sets[s]]
        frames = np.array(frames)
        with open('%sset_labels'%(setname), 'r') as f:
            labels = f.read()
        labels = np.array([int(l.split('\t')[1]) for l in labels.split('\n')[:-1]])
        writer = tf.python_io.TFRecordWriter('%s.tfrecords'%(setname))
        rows = frames.shape[1]
        cols = frames.shape[2]
        for i in range(frames.shape[0]):
            image_raw = images[i].tostring()
            label_raw = labels[i].tostring()
            data = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'label_raw': _bytes_feature(label_raw),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(data.SerializeToString())
        writer.close()
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments for the dataset spliter')
    parser.add_argument('--dataset', dest='dataset', default='MiniDrone', help='Path to the dataset to split')
    parser.add_argument('--format', dest='format', default='mp4', help='Videos format')
    parser.add_argument('--subset', dest='subset', default='train,test', help='Subsets for the dataset to be divided in')
    args = parser.parse_args()
    main(args)
