import glob
from sklearn.model_selection import StratifiedKFold
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_splits', type = int)
parser.add_argument('--pos_path', type = str, help = 'Path containing positive examples frames')
parser.add_argument('--neg_path', type = str, help = 'Path containing negative examples frames')
args = parser.parse_args()

def main(num_splits, pos_path, neg_path)
    pos = glob.glob('{}/*'.format(pos_path))
    neg = glob.glob('{}/*'.format(neg_path))

    x = pos + neg
    y = [1] * len(pos) + [0] * len(neg)

    X = ['{} {}'.format(b, len(glob.glob('{}/img*'.format(b)))) for b in x]

    X = np.array(X)
    y = np.array(y)

    kf = StratifiedKFold(num_splits)
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

if __name__ == '__main__':
    main(args.num_splits, args.pos_path, args.neg_path)