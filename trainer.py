import tensorflow as tf
import argparse

import vgg16

def main(args):
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments for the model\'s trainer')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--ftrain', dest='trainable', type=bool, default=False, help='Full train (VGG)')
    parser.add_argument('--dsummary', dest='dataset_summary', type=str, default='', help='Path to the dataset summary')
    parser.add_argument('--trsummary', dest='dataset_summary', type=str, default='', help='Path to the train set summary')
    parser.add_argument('--vsummary', dest='dataset_summary', type=str, default='', help='Path to the val set summary')
    parser.add_argument('--tesummary', dest='dataset_summary', type=str, default='', help='Path to the test set summary')
    args = parser.parse_args()
    main(args)
