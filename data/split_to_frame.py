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
    #Directory for resulting frames
    if not os.path.exists(args.frames_path):
        os.makedirs(args.frames_path)
    #Gather videos
    subsets = args.subset.split(',')
    sets = [[p + '/' + video for p, n, f in os.walk(args.dataset) for video in f if s in p and args.format in video] for s in subsets]
    #Split the set's videos frame by frame
    for s in range(len(sets)):
        setname = subsets[s]
        print('Split %sset...'%(setname))
        if not os.path.exists(args.frames_path + '/%sset'%(setname)):
            os.makedirs(args.frames_path + '/%sset'%(setname))
        for video in tqdm(sets[s]):
            v = imageio.get_reader(video, 'ffmpeg')
            nb_frames = v.get_meta_data()['nframes']
            for f in range(nb_frames):
                #Sometimes a RuntimeError occurs while fetching the last frame
                try:
                    f_data = v.get_data(f)
                    if args.resize is not '':
                        resize = [int(dim) for dim in args.resize.split(',')]
                        scipy.misc.imsave('%s/%sset/%s_%d.png'%(args.frames_path, setname, ntpath.basename(video)[:-4], f + 1), scipy.misc.imresize(f_data, (resize[0], resize[1])))
                    else:
                        scipy.misc.imsave('%s/%sset/%s_%d.png'%(args.frames_path, setname, ntpath.basename(video)[:-4], f + 1), f_data)
                except RuntimeError:
                    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments for the dataset spliter')
    parser.add_argument('--dataset', dest='dataset', default='MiniDrone', help='Path to the dataset to split')
    parser.add_argument('--format', dest='format', default='mp4', help='Videos format')
    parser.add_argument('--frames', dest='frames_path', default='MiniDrone_frames', help='Path to the directory used to store the resulting frames')
    parser.add_argument('--resize', dest='resize', default='224,224', help='Specify the size at which the frames should be resized (format: heigt,width)')
    parser.add_argument('--subset', dest='subset', default='train,val,test', help='Subsets for the dataset to be divided in (format: subset1,subset2,...,subsetn)')
    args = parser.parse_args()
    main(args)
