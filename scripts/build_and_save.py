import argparse
import tensorflow as tf
#from keras.utils import plot_model

import modules.build_network as bn

def parse_args():
    parser = argparse.ArgumentParser(description='Build and save model.')
    parser.add_argument('--out', help='Output H5 model file.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    
    model = bn.build_network()
    model.save(args.out)

    model.summary()
