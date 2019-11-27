import argparse
import json
import numpy as np
import random
import modules.utils as utils
import scipy.io as sio
import tensorflow.keras as keras
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--model', help='Path to the model to train.', required=True)
    parser.add_argument('--data', help='Path to data directory.', required=True)
    parser.add_argument('--trained_model', help='Path to save trained model.', required=True)    
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    model = keras.models.load_model(args.model)
    
    file_path = os.path.join(args.data, 'eval_data.npy')
    eval_data = np.load(file_path)
    file_path = os.path.join(args.data, 'eval_labels.npy')
    eval_labels = np.load(file_path)
    file_path = os.path.join(args.data, 'train_data.npy')
    train_data = np.load(file_path)
    file_path = os.path.join(args.data, 'train_labels.npy')
    train_labels = np.load(file_path)

    model_dir = os.path.dirname(args.model)
    
    stopping = keras.callbacks.EarlyStopping(patience=12)
    checkpoints = keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), verbose = 1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.1,
        patience=2,
        min_lr=0.001 * 0.001)
    batch_size = 12
    model.fit(train_data, train_labels,
          batch_size = batch_size,
          shuffle = True,
          epochs=100,
          validation_data=(eval_data, eval_labels),
          callbacks = [stopping, reduce_lr, checkpoints])

    model.save(args.trained_model)
