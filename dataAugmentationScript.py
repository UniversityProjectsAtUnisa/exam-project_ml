import numpy as np
import h5py
import math
from collections import Counter, defaultdict
from utils import one_hot, de_one_hot


def augment_sample(elem: np.ndarray, wanted_samples=1):
    if wanted_samples < 1:
        return np.array([elem])

    if wanted_samples == 1:
        return np.array([elem])

    elems_augmented = [elem]

    total = (elem.shape[0])
    slice_amounts = [(i * total//(wanted_samples+1))
                     for i in range(1, wanted_samples+1)]

    for slice_amount in slice_amounts:
        elems_augmented.append(
            np.concatenate((elem[slice_amount:], elem[:slice_amount]), axis=0))

    assert len(elems_augmented) == wanted_samples
    return np.array(elems_augmented)


def augment_samples(arr: np.ndarray, wanted_samples=0):
    if wanted_samples < len(arr):
        return arr

    X_augmented = []
    total_augmentation = wanted_samples

    for i, elem in enumerate(arr):
        current_augmentation = math.ceil(total_augmentation / (len(arr) - i))
        augmented_elements = augment_sample(elem, current_augmentation)
        X_augmented.extend(list(augmented_elements))
        total_augmentation -= current_augmentation
    assert len(X_augmented) == wanted_samples
    return np.array(X_augmented)


def augment_data(data, results):
    unhot_results = de_one_hot(results)
    c = Counter(unhot_results)
    wanted = next(v for v in c.values())

    augmented = []
    augmented_results = []
    to_augment = defaultdict(list)

    for ind, elem in enumerate(data):
        elem_class = unhot_results[ind]
        if c[elem_class] == wanted:
            augmented.append(elem)
            augmented_results.append(elem_class)
        else:
            to_augment[elem_class].append(elem)

    for elem_class, elems in to_augment.items():
        augmented.extend(list(augment_samples(elems, wanted)))
        augmented_results.extend([elem_class]*wanted)

    return np.array(augmented), one_hot(augmented_results, results.shape[1])


file_h5 = './data/pre_augment_train.h5'
f = h5py.File(file_h5, 'r')
X = f['X'][...]
y = f['y'][...]
f.close()


augmented_data, augmented_results = augment_data(X, y)

file_h5 = './data/augmented_train.h5'
f = h5py.File(file_h5, 'w')
f.create_dataset('X', data=augmented_data)
f.create_dataset('y', data=augmented_results)
f.close()
