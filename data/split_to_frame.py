"""
Split each videos frame by frame.
"""

import argparse
import os
import ntpath
import imageio
import scipy.misc
from tqdm import tqdm

def main(args):
    """
    """

    if not os.path.exists(args.frames_path):
        os.makedirs(args.frames_path)
    #Gather videos
    train_set = [p + '/' + video for p, n, f in os.walk(args.dataset) for video in f if 'train' in p and args.format in video]
    test_set = [p + '/' + video for p, n, f in os.walk(args.dataset) for video in f if 'test' in p and args.format in video]
    #Split frame by frame
    print('Split train set...')
    if not os.path.exists(args.frames_path + '/trainset'):
        os.makedirs(args.frames_path + '/trainset')
    for video in tqdm(train_set):
        v = imageio.get_reader(video, 'ffmpeg')
        nb_frames = v.get_meta_data()['nframes']
        for f in range(nb_frames):
            try:
                f_data = v.get_data(f)
                if args.resize != '':
                    resize = [int(dim) for dim in args.resize.split(',')]
                    scipy.misc.imsave('%s/trainset/%s_%d.png'%(args.frames_path, ntpath.basename(video)[:-4], f), scipy.misc.imresize(f_data, (resize[0], resize[1])))
                else:
                    scipy.misc.imsave('%s/trainset/%s_%d.png'%(args.frames_path, ntpath.basename(video)[:-4], f), f_data)
            except RuntimeError:
                pass
    print('Split test set...')
    if not os.path.exists(args.frames_path + '/testset'):
        os.makedirs(args.frames_path + '/testset')
    for video in tqdm(test_set):
        v = imageio.get_reader(video, 'ffmpeg')
        nb_frames = v.get_meta_data()['nframes']
        for f in range(nb_frames):
            try:
                f_data = v.get_data(f)
                if args.resize != '':
                    resize = [int(dim) for dim in args.resize.split(',')]
                    scipy.misc.imsave('%s/testset/%s_%d.png'%(args.frames_path, ntpath.basename(video)[:-4], f), scipy.misc.imresize(f_data, (resize[0], resize[1])))
                else:
                    scipy.misc.imsave('%s/testset/%s_%d.png'%(args.frames_path, ntpath.basename(video)[:-4], f), f_data)
            except RuntimeError:
                pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments for the dataset spliter')
    parser.add_argument('--dataset', dest='dataset', default='MiniDrone', help='Path to the dataset to split')
    parser.add_argument('--format', dest='format', default='mp4', help='Videos format')
    parser.add_argument('--frames', dest='frames_path', default='frames', help='Path to the directory used to store the resulting frames')
    parser.add_argument('--resize', dest='resize', default='', help='Specify the size at which the frames should be resized (format: heigt,width)')
    args = parser.parse_args()
    main(args)
