from collections import Counter
import h5py
import numpy as np
import utils


def test_dimension(filename):
    f = h5py.File(filename, 'r')
    y = f['y'][...]
    f.close()

    c = Counter(utils.de_one_hot(y))
    print(c)


test_dimension('./data/augmented_train.h5')
# test_dimension('./data/test.h5')
