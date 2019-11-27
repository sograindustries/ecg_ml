import json
import numpy as np
import scipy.io as sio
import os
import tensorflow.keras as keras

LABEL_MAP = {'N': 0, 'A': 1, 'O': 1, '~': 1}
LABEL_MAP_FULL = {'N': 0, 'A': 1, 'O': 2, '~': 3}

def generate_training_data(csv_path):
    reference_path = csv_path
    with open(reference_path, 'r') as f:
        labels = [line.strip().split(',') for line in f]
    ecg_data = []
    ecg_labels_i = []
    for item in labels:
        mat_file_path = './data/training2017/' + item[0] + '.mat'
        ecg = sio.loadmat(mat_file_path)['val'].squeeze()
        if ecg.shape[0] != 9000:
            continue
        ecg_data += [ecg[:8960]]
        ecg_labels_i += [[LABEL_MAP_FULL[item[1]]]*35]
    ecg_labels = [[keras.utils.to_categorical(ii, 4) for ii in ss] for ss in ecg_labels_i]
    mean = np.mean(ecg_data)
    std = np.std(ecg_data)
    ecg_data = (ecg_data - mean)/std
    ecg_labels = np.array(ecg_labels)
    return ecg_data, ecg_labels
        
def load_cinc_data(data_path, json_path):
    labels = []
    data = []
    with open(json_path, 'r') as f:
        label_and_data = [json.loads(line) for line in f]
    for item in label_and_data:
        _l = LABEL_MAP[item['label']]
        label_temp = [0, 0]
        label_temp[_l] = 1
        labels += [label_temp]
        data_path_sample = os.path.join(data_path, item['path'])
        data += [sio.loadmat(data_path_sample)['val'].squeeze()]
    return np.expand_dims(np.array(data).astype(np.float32), 2), np.array(labels)
