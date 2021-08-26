import os,sys

import argparse

parser = argparse.ArgumentParser(description='Download data.')
parser.add_argument('--data', type=str, help='mnist, celeba')
parser.add_argument('--data_dir', type=str, help='download location')


sys.path.append('..')


from bridge.data.stackedmnist import Stacked_MNIST
from bridge.data.emnist import EMNIST
from bridge.data.celeba  import CelebA


# SETTING PARAMETERS

def main():

    args = parser.parse_args()

    if args.data == 'mnist':
        root = os.path.join(args.data_dir, 'mnist')
        Stacked_MNIST(root, 
                      load=False, 
                      source_root=root,
                      train=True, 
                      num_channels = 1,
                      imageSize=28,
                      device='cpu')

    if args.data == 'celeba':
        root = os.path.join(args.data_dir, 'celeba')
        CelebA(root, split='train', download=True)
    

if __name__ == '__main__':
    main()  
