import argparse
import os

def main(args):
    sets = args.subset.split(',')
    for s in sets:
        with open('%sset_labels'%(s), 'r') as f:
            content = f.read()
        content = [c.split('\t') for c in content.split('\n')[:-1]]
        frames = [p + '/' + frame for p, n, f in os.walk(args.dataset) for frame in f if s in p and args.format in frame]
        to_remove = []
        for l in content:
            if (l[0] + '.png') not in frames:
                to_remove.append(l)
        new = list(content)
        for r in to_remove:
            print('removing %s...'%(r))
            new.remove(r)
        with open('%sset_labels'%(s), 'w') as f:
            for n in new:
                f.write(n[0] + '\t' + n[1] + '\t' + n[2] + '\n')
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments for the dataset spliter')
    parser.add_argument('--dataset', dest='dataset', default='MiniDrone_frames', help='Path to the dataset to split')
    parser.add_argument('--format', dest='format', default='png', help='Videos format')
    parser.add_argument('--frames', dest='frames_path', default='MiniDrone_frames', help='Path to the directory used to store the resulting frames')
    parser.add_argument('--resize', dest='resize', default='224,224', help='Specify the size at which the frames should be resized (format: heigt,width)')
    parser.add_argument('--subset', dest='subset', default='train,val,test', help='Subsets for the dataset to be divided in (format: subset1,subset2,...,subsetn)')
    args = parser.parse_args()
    main(args)
