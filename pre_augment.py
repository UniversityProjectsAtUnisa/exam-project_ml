from functools import partial
import h5py
import utils
import numpy as np
from collections import Counter, defaultdict

N_CLASSES = 55


# Load data from professor training set
file_h5 = './data/train.h5'
f = h5py.File(file_h5, 'r')
X = f['X'][...]
y = f['y'][...]
f.close()


# Split data in 3 groups
unhot_y = utils.de_one_hot(y)

c = Counter()
test_set = []
test_results = []
preaugment_training_set = defaultdict(list)
partial_x_train = []
partial_y_train = []


# Get the test data
for i, elem in enumerate(X):
    res = unhot_y[i]
    if c[res] < 25:
        test_set.append(elem)
        test_results.append(y[i])
        c[res] += 1
    else:
        # Temporarily store data after removing test data
        preaugment_training_set[res].append(elem)
        c[res] += 1

max_class_occurrences = c.most_common(1)[1]

# Separate ready training set from the data that needs to be augmented first
for k, v in c.items():
    if v < max_class_occurrences:
        temp = preaugment_training_set.pop(k)
        partial_x_train.append(temp)
        partial_y_train.append(
            [utils.one_hot(k, N_CLASSES)]*max_class_occurrences)


# Store ready-to-go test data
file_h5 = './data/test.h5'
with h5py.File(file_h5, 'w') as f:
    f.create_dataset('X_test', data=test_set)
    f.create_dataset('y_test', data=test_results)


file_h5 = './data/preaugment_training_set.h5'
with h5py.File(file_h5, 'w') as f:
    f.create_dataset('preaugment', data=test_set)
    f.create_dataset('y_test', data=test_results)
    augment_target = 425


file_h5 = './data/partial_training_set.h5'
with h5py.File(file_h5, 'w') as f:
    f.create_dataset('X', data=partial_x_train)
    f.create_dataset('y', data=partial_y_train)
