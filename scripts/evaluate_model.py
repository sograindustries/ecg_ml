import argparse
import json
import numpy as np
import random
import modules.utils as utils
import scipy.io as sio
import tensorflow.keras as keras
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a model.')
    parser.add_argument('--model', help='Path to the model.', required=True)
    parser.add_argument('--data', help='Path to data directory.', required=True)
    parser.add_argument('--out', help='Path to save collateral.', required=True)    
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    model = keras.models.load_model(args.model)
    
    file_path = os.path.join(args.data, 'eval_data.npy')
    eval_data = np.load(file_path)
    file_path = os.path.join(args.data, 'eval_labels.npy')
    eval_labels = np.load(file_path)

    predict_labels = model.predict(eval_data)
    np.save(os.path.join(args.out, 'predict.npy'), predict_labels)
    
    fpr, tpr, thresholds = roc_curve(eval_labels[:, 0], predict_labels[:, 0])
    np.save(os.path.join(args.out, 'fpr.npy'), fpr)
    np.save(os.path.join(args.out, 'tpr.npy'), tpr)
    np.save(os.path.join(args.out, 'thresholds.npy'), thresholds)        
    
    auc = roc_auc_score(eval_labels[:, 0], predict_labels[:, 0])
    print('Area under curve: {}'.format(auc))

    # Finds examples of fp, fn, tp, tn
    tn = []
    tn_i = []
    tp = []
    tp_i = []
    fn = []
    fn_i = []
    fp = []
    fp_i = []
    print(eval_labels.shape)
    print(predict_labels.shape)
    print(eval_data.shape)
    for index, data in enumerate(zip(eval_labels, predict_labels)):
        el = data[0]
        pl = data[1]
        p = np.argmax(pl)
        e = np.argmax(el)
        if (p == 0 and p == e):
            tn += [eval_data[index, :]]
            tn_i += [index]
        if (p == 1 and p == e):
            tp += [eval_data[index, :]]
            tp_i += [index]
        if (p == 0 and p != e):
            fn += [eval_data[index, :]]
            fn_i += [index]
        if (p == 1 and p != e):
            fp += [eval_data[index, :]]
            fp_i += [index]

    path = os.path.join(args.out, 'tn.npy')
    np.save(path, tn)
    path = os.path.join(args.out, 'tn_i.npy')
    np.save(path, tn_i)

    path = os.path.join(args.out, 'tp.npy')
    np.save(path, tp)
    path = os.path.join(args.out, 'tp_i.npy')
    np.save(path, tp_i)

    path = os.path.join(args.out, 'fn.npy')
    np.save(path, fn)
    path = os.path.join(args.out, 'fn_i.npy')
    np.save(path, fn_i)
    
    path = os.path.join(args.out, 'fp.npy')
    np.save(path, fp)
    path = os.path.join(args.out, 'fp_i.npy')
    np.save(path, fp_i)

    
