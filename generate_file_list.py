
import glob
import os
from sklearn.model_selection import StratifiedKFold
import numpy as np

pos = glob.glob('frames/aggressive/*')
neg = glob.glob('frames/passive_cctv/*')

x = pos + neg
y = [1] * len(pos) + [0] * len(neg)

X = ['{} {}'.format(b, len(glob.glob('{}/img*'.format(b)))) for b in x]

X = np.array(X)
y = np.array(y)

kf = StratifiedKFold(3)
for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    X_train = X[train_idx]
    y_train = y[train_idx]
    
    with open('data/cctv_train_split_{}.txt'.format(i), 'w') as fh:
        for j in range(len(X_train)):
            fh.write('{} {}\n'.format(X_train[j], y_train[j]))
            
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    with open('data/cctv_test_split_{}.txt'.format(i), 'w') as fh:
        for j in range(len(X_test)):
            fh.write('{} {}\n'.format(X_test[j], y_test[j]))





