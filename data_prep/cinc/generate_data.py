import argparse
import json
import numpy as np
import random
import scipy.io as sio
import scipy.signal as signal
import tensorflow.keras as keras
import os

SEGMENT_LENGTH = 128

LABEL_MAP_BINARY = {'N': 0, 'A': 1, 'O': 1, '~': 1}

def parse_args():
    parser = argparse.ArgumentParser(description='Generate training and evaluation json.')
    parser.add_argument('--out', help='Output directory to place data.', required=True)
    parser.add_argument('--dir', help='Path to CINC directory.', required=True)
    parser.add_argument('--cutoff', help='Cutoff value to use', default=0.1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    
    # Loads the reference CSV file
    csv_path = os.path.join(args.dir, 'REFERENCE.csv')
    with open(csv_path, 'r') as f:
        labels = [line.strip().split(',') for line in f]

    # Shuffles data
    random.shuffle(labels)
    
    ecg_data = []
    ecg_labels_i = []
    data_path = args.dir
    for item in labels:
        mat_file_path = os.path.join(data_path, item[0] + '.mat')
        ecg = sio.loadmat(mat_file_path)['val'].squeeze()
        if ecg.shape[0] < 9000:
            continue
        # Resamples data from 300 Hz to 250 Hz.
        resampled_ecg = signal.resample(ecg[:8909], 7424)
        ecg_data += [resampled_ecg]
        ecg_labels_i += [[LABEL_MAP_BINARY[item[1]]]]        
    ecg_labels = ecg_labels_i
    ecg_labels = np.array(ecg_labels)
    print(ecg_labels.shape)
    ecg_data = np.array(ecg_data)    

    # Expand dims
    ecg_data = np.expand_dims(ecg_data, 2)

    for ii in range(ecg_data.shape[0]):
        mean = np.mean(ecg_data[ii,:,0])
        std = np.std(ecg_data[ii,:,0])
        ecg_data[ii,:,0] = np.array((ecg_data[ii, :, 0] - mean)/std)
    ecg_labels = np.array(ecg_labels)
    
    # Creates evaluation and training data sets.
    cutoff = int(np.round(ecg_data.shape[0]*args.cutoff))
    eval_data = ecg_data[:cutoff, :, :]
    eval_labels = ecg_labels[:cutoff, :]
    train_data = ecg_data[cutoff:, :, :]
    train_labels = ecg_labels[cutoff:, :]

    # Saves outputs as numpy files
    file_path = os.path.join(args.out, 'eval_data.npy')
    np.save(file_path, eval_data)
    file_path = os.path.join(args.out, 'eval_labels.npy')
    np.save(file_path, eval_labels)
    file_path = os.path.join(args.out, 'train_data.npy')
    np.save(file_path, train_data)
    file_path = os.path.join(args.out, 'train_labels.npy')
    np.save(file_path, train_labels)
    print(eval_labels.shape)
