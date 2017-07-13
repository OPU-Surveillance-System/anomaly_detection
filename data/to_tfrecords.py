"""
Save specified data and their label into tfrecord files.
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from scipy import misc
from tqdm import tqdm

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main(args):
    #Create tfrecord writer
    writer = tf.python_io.TFRecordWriter('%s.tfrecord'%(args.record_name))
    #Parse dataset summary (file listing images and their label)
    with open(args.set_summary, 'r') as f:
        summary = f.read()
    summary = summary.split('\n')[:-1]
    summary = [l.split('\t') for l in summary]
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
    for s in tqdm(summary):
        #Slide a window over frames and write content in the tfrecord
        for w in range(len(s) - (args.window_lenght - 1)):
            #Load frames
            frames = [misc.imread('%s.png'%(f[0]), mode='L') for f in s[w:w + args.window_lenght]]
            #Apply transformation if requested
            if args.data_modification == 'flip':
                frames = [np.fliplr(f) for f in frames]
            frames = np.array(frames)
            frames = np.transpose(frames, (1, 2, 0))
            #Get frames' height width
            rows = frames.shape[0]
            cols = frames.shape[1]
            #Load label
            label = sum([int(l[1]) for l in s[w:w + args.window_lenght]])
            if label > 1:
                label = 1
            #Write to tfrecord file
            frames_raw = frames.tostring()
            data = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'label': _int64_feature(label),
                'image': _bytes_feature(frames_raw)}))
            writer.write(data.SerializeToString())
    writer.close()

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments for the dataset spliter')
    parser.add_argument('--summary', dest='set_summary', type=str, default='trainset_labels', help='Path to file summarizing the dataset')
    parser.add_argument('--wl', dest='window_lenght', type=int, default=1, help='Lenght of the slinding window over frames')
    parser.add_argument('--modif', dest='data_modification', type=str, default=None, help='Data augmentation method (None, flip)')
    parser.add_argument('--name', dest='record_name', type=str, default='trainset', help='Name for the resulting tf record file')
    args = parser.parse_args()
    main(args)
