"""
"""

import argparse
import os
import xml.etree.ElementTree as etree
from tqdm import tqdm

def main(args):
    anomalies = {
        "walking":0,
        "fighting": 1,
        "picking_up": 1,
        "standing": 0,
        "attacking": 1,
        "talking": 0,
        "stealing": 1,
        "cycling": 0,
        "loitering": 1,
        "running": 0,
        "driving": 1,
        "parking": 0,
        "parked": 0,
        "moving": 0,
        "stopping": 0,
        "falling": 1,
        "repairing": 1,
        "dropping_off": 1,
        "throwing_away": 1
    }

    misc = {
        "SkinRegion",
        "Face",
        "Hair",
        "Accessory",
        "LicensePlate"
    }
    #Test if the path to the frames is valid
    assert os.path.exists(args.frames_path), 'The frames\' path is not valid'
    #Gather labels
    train_set = [p + '/' + label for p, n, f in os.walk(args.dataset) for label in f if 'train' in p and args.format in label]
    test_set = [p + '/' + label for p, n, f in os.walk(args.dataset) for label in f if 'test' in p and args.format in label]
    #Label trainset
    print('Labeling trainset...')
    for l in tqdm(train_set):
        data_name = l.split('/')[2][:-5]
        data_description = etree.parse(l).getroot()[1][0]
        objects = [o for o in data_description if 'object' in o.tag and o.attrib['name'] not in misc]
        actions = [[act for attr in o for act in attr if attr.attrib['name'] == 'Action'] for o in objects]
        numframes = int(data_description[0][1][0].attrib['value'])
        data = {i:[] for i in range(1, numframes + 1)}
        for obj in actions:
            for act in obj:
                val = act.attrib['value']
                framespan = [int(f) for f in act.attrib['framespan'].split(':')]
                for frame in range(framespan[0], framespan[1] + 1):
                    try:
                        data[frame] += [val]
                    except KeyError:
                        data[frame] = [val]
        with open('trainset_labels', 'a') as f:
            for frame in sorted(data.keys()):
                label = sum([anomalies[elt] for elt in data[frame]])
                if label > 1:
                    label = 1
                if len(data[frame]) == 0:
                    data[frame] = ['nothing']
                f.write('%s/trainset/%s_%d\t%d\t%s\n'%(args.frames_path, data_name, frame, label, ','.join(data[frame])))
    with open('trainset_labels', 'r') as f:
        content = f.read().split('\n')[:-1]
    print(len(content))
    content = [c.split("\t")[0].split("/")[2] for c in content]
    train_set = []
    for p, n, f in os.walk(args.dataset + "_frames/trainset"):
        for files in f:
            train_set.append(files[:-4])
    for t in train_set:
        if t not in content:
            print(t)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments for the dataset labeler')
    parser.add_argument('--dataset', dest='dataset', default='MiniDrone', help='Path ground truth labels')
    parser.add_argument('--format', dest='format', default='xgtf', help='Labels format')
    parser.add_argument('--frames', dest='frames_path', default='MiniDrone_frames', help='Path to the frames to label')
    args = parser.parse_args()
    main(args)
