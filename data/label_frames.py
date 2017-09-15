"""
Label a set of frames given an associated set of descriptor (xml/GtViper...).
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
        "cycling": 1,
        "loitering": 1,
        "running": 1,
        "driving": 0,
        "parking": 0,
        "parked": 0,
        "moving": 0,
        "stopping": 0,
        "falling": 1,
        "repairing": 1,
        "dropping_off": 1,
        "throwing_away": 1,
        "riding": 0,
        "intimidating": 1,
        "reserving": 1
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
    subsets = args.subset.split(',')
    sets = [[p + '/' + label for p, n, f in os.walk(args.dataset) for label in f if s in p and args.format in label] for s in subsets]
    #Label sets
    for s in range(len(subsets)):
        setname = subsets[s]
        with open('%sset_labels'%(setname), 'w') as f:
            print('Cleaning previous {}set_labels'.format(setname))
        print('Labeling %sset...'%(setname))
        for l in tqdm(sets[s]):
            data_name = l.split('/')[2][:-5]
            #Get descriptor root
            data_description = etree.parse(l).getroot()[1][0]
            #Get objects (person, car, ...)
            objects = [o for o in data_description if 'object' in o.tag and o.attrib['name'] not in misc]
            #Get objects' actions
            actions = [[act for attr in o for act in attr if attr.attrib['name'] == 'Action'] for o in objects]
            #Get number of frames
            numframes = int(data_description[0][1][0].attrib['value'])
            data = {i:[] for i in range(1, numframes + 1)}
            #For each frames indicates the action of all objects
            for obj in actions:
                for act in obj:
                    val = act.attrib['value']
                    framespan = [int(f) for f in act.attrib['framespan'].split(':')]
                    for frame in range(framespan[0], framespan[1] + 1):
                        try:
                            data[frame] += [val]
                        except KeyError:
                            data[frame] = [val]
            #Associate a label (0 or 1) to each frame
            with open('%sset_labels'%(setname), 'a') as f:
                for frame in sorted(data.keys()):
                    label = sum([anomalies[elt] for elt in data[frame]])
                    if label > 1:
                        label = 1
                    if len(data[frame]) == 0:
                        data[frame] = ['nothing']
                    if os.path.isfile('{}/{}set/{}-{}.png'.format(args.frames_path, setname, data_name, frame)):
                        f.write('%s/%sset/%s-%d\t%d\t%s\n'%(args.frames_path, setname, data_name, frame, label, ','.join(data[frame])))
                    # else:
                    #     print('**{}/{}set/{}-{}.png not found**'.format(args.frames_path, setname, data_name, frame))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments for the dataset labeler')
    parser.add_argument('--dataset', dest='dataset', default='MiniDrone', help='Path ground truth labels')
    parser.add_argument('--format', dest='format', default='xgtf', help='Labels format')
    parser.add_argument('--frames', dest='frames_path', default='MiniDrone_frames', help='Path to the frames to label')
    parser.add_argument('--subset', dest='subset', default='train,val,test', help='Subsets in which the dataset is divided in (format: subset1,subset2,...,subsetn)')
    args = parser.parse_args()
    main(args)
