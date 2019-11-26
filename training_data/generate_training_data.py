import numpy as np
import tensorflow.keras as keras

# CINC Data
cinc_eval_data = np.load('../data_prep/cinc/data/eval_data.npy')
cinc_eval_labels = np.load('../data_prep/cinc/data/eval_labels.npy')
cinc_train_data = np.load('../data_prep/cinc/data/train_data.npy')
cinc_train_labels = np.load('../data_prep/cinc/data/train_labels.npy')

# MIT Data
mit_eval_data = np.load('../data_prep/mit-bih/data/eval_data.npy')
mit_eval_labels = np.load('../data_prep/mit-bih/data/eval_labels.npy')
mit_train_data = np.load('../data_prep/mit-bih/data/train_data.npy')
mit_train_labels = np.load('../data_prep/mit-bih/data/train_labels.npy')

# Patient Simulator Data
ps_eval_data = np.load('../data_prep/patient_simulator/data/eval_data.npy')
ps_eval_labels = np.load('../data_prep/patient_simulator/data/eval_labels.npy')
ps_train_data = np.load('../data_prep/patient_simulator/data/train_data.npy')
ps_train_labels = np.load('../data_prep/patient_simulator/data/train_labels.npy')

eval_data = np.concatenate((cinc_eval_data, mit_eval_data, ps_eval_data), 0)
eval_labels = np.concatenate((cinc_eval_labels, mit_eval_labels, ps_eval_labels), 0)
train_data = np.concatenate((cinc_train_data, mit_train_data, ps_train_data), 0)
train_labels = np.concatenate((cinc_train_labels, mit_train_labels, ps_train_labels), 0)

print('Training data size: {}x{}x{}'.format(*train_data.shape))
print('Eval data size: {}x{}x{}'.format(*eval_data.shape))

eval_labels_cat = np.zeros((eval_labels.shape[0], 2))
for ii in range(eval_labels_cat.shape[0]):                         
    eval_labels_cat[ii, :] = keras.utils.to_categorical(eval_labels[ii], 2)

train_labels_cat = np.zeros((train_labels.shape[0], 2))
for ii in range(train_labels_cat.shape[0]):                         
    train_labels_cat[ii, :] = keras.utils.to_categorical(train_labels[ii], 2)

# Normalizes data by mean and std. dev.
for ii in range(eval_data.shape[0]):
    eval_data[ii, :, 0] = (eval_data[ii, :, 0] - np.mean(eval_data[ii, :, 0])) / np.std(eval_data[ii, :, 0])
for ii in range(train_data.shape[0]):
    train_data[ii, :, 0] = (train_data[ii, :, 0] - np.mean(train_data[ii, :, 0])) / np.std(train_data[ii, :, 0])

    
np.save('./eval_data', eval_data)
np.save('./eval_labels', eval_labels_cat)
np.save('./train_data', train_data)
np.save('./train_labels', train_labels_cat)
