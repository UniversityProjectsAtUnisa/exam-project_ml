import utils
from collections import Counter, defaultdict
import h5py

file_h5 = './data/train.h5'
f = h5py.File(file_h5, 'r')
X = f['X'][...]
y = f['y'][...]
f.close()


def split_data_sets(X, y, n=25):
    """Split training set from test set by keeping the test set balanced

    Args:
        X (ndarray): Data sets
        y (ndarray): Data classes
        n (int) Target test set dimension
    """
    unhot_y = utils.de_one_hot(y)

    c = Counter()
    test_set = []
    test_results = []
    training_set = []
    training_results = []

    for i, elem in enumerate(X):
        res = unhot_y[i]
        if c[res] < n:
            test_set.append(elem)
            test_results.append(y[i])
        else:
            training_set.append(elem)
            training_results.append(y[i])
        c[res] += 1

    return test_set, test_results, training_set, training_results


X_test, y_test, X_traning, y_training = split_data_sets(X, y, 25)


file_h5 = './data/pre_augment_train.h5'
f = h5py.File(file_h5, 'w')
f.create_dataset('X', data=X_traning)
f.create_dataset('y', data=y_training)
f.close()

file_h5 = './data/test.h5'
f = h5py.File(file_h5, 'w')
f.create_dataset('X', data=X_test)
f.create_dataset('y', data=y_test)
f.close()
