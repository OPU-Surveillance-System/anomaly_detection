"""
Split each videos frame by frame.
"""

import os
import imageio
from scipy import misc
from tqdm import tqdm

names = {
    '1_1': (0, 624),
    '1_2': (625, 1452),
    '2_1': (1453, 2001),
    '2_2': (2002, 2686),
    '2_3': (2687, 3454),
    '2_4': (3455, 4033),
    '2_5': (4034, 4928),
    '2_6': (4929, 5595),
    '3_1': (5596, 6253),
    '3_2': (6254, 6930),
    '3_3': (6931, 7738)
}

if not os.path.exists('UMN_frames'):
    os.makedirs('UMN_frames')
v = imageio.get_reader('UMN/Crowd-Activity-All.avi', 'ffmpeg')
nb_frames = v.get_meta_data()['nframes']
for f in tqdm(range(nb_frames)):
    try:
        f_data = v.get_data(f)
        misc.imsave('UMN_frames/frame_{}.png'.format(f), misc.imresize(f_data, (224, 224)))
    except RuntimeError:
        pass
for k in tqdm(list(names.keys())):
    for i in range(names[k][0], names[k][1] + 1):
        os.rename('UMN_frames/frame_{}.png'.format(i),
                  'UMN_frames/frame_{}_{}.png'.format(k, i))
