from collections import Counter
import numpy as np
import h5py
import random
import math
from collections import Counter, defaultdict
from profiler import profile
"""
PADDING = 8

def augment_sample(elem: np.ndarray, wanted_samples=1):
    if wanted_samples < 1:
        return np.array([elem])

    if wanted_samples == 1:
        return np.array([elem])

    elems_augmented = [elem]

    slice_amounts = random.sample(
        range(PADDING, elem.shape[0]-PADDING), k=wanted_samples-1)

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
 """

from collections.abc import Sequence

PADDING = 8


# @profile()
def get_shape(lst, shape=()):
    """
    returns the shape of nested lists similarly to numpy's shape.

    :param lst: the nested list
    :param shape: the shape up to the current recursion depth
    :return: the shape including the current depth
            (finally this will be the full depth)
    """

    if not isinstance(lst, Sequence):
        # base case
        return shape

    # peek ahead and assure all lists in the next depth
    # have the same length
    if isinstance(lst[0], Sequence):
        l = len(lst[0])
        if not all(len(item) == l for item in lst):
            msg = 'not all lists have the same length'
            raise ValueError(msg)

    shape += (len(lst), )

    # recurse
    shape = get_shape(lst[0], shape)

    return shape


# @profile()
def augment_sample(elem: list, wanted_samples=1):
    if wanted_samples < 1:
        return [elem]

    if wanted_samples == 1:
        return [elem]

    elems_augmented = [elem]

    slice_amounts = random.sample(
        range(PADDING, get_shape(elem)[0]-PADDING), k=wanted_samples-1)

    for slice_amount in slice_amounts:
        new_elem = elem[slice_amount:]
        new_elem.extend(elem[:slice_amount])
        elems_augmented.append(
            new_elem)
        assert len(new_elem) == len(elem)

    assert len(elems_augmented) == wanted_samples
    return elems_augmented


@profile()
def augment_samples(arr: list, wanted_samples=0):
    if wanted_samples < len(arr):
        return arr

    X_augmented = []
    total_augmentation = wanted_samples

    for i, elem in enumerate(arr):
        current_augmentation = math.ceil(total_augmentation / (len(arr) - i))
        augmented_elements = augment_sample(elem, current_augmentation)
        X_augmented.extend(augmented_elements)
        total_augmentation -= current_augmentation
    assert len(X_augmented) == wanted_samples
    return X_augmented


# @profile()
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
            # Mi voglio salvare solo gli indici in cui ho bisogno di fare augmentation
            to_augment[elem_class].append(ind)

    for elem_class, elems_ind in to_augment.items():
        x = augment_samples([e.tolist() for e in data[elems_ind]], wanted)
        print(get_shape(x))
        np.append(data, np.array(x), axis=0)
        np.append(results, np.array([elem_class*wanted]))

    return np.array(data), one_hot(results, results.shape[1])


file_h5 = './data/train.h5'
f = h5py.File(file_h5, 'r')
X = f['X'][...]
y = f['y'][...]
f.close()


def one_hot(a, n):
    e = np.eye(n)  # Identity matrix n x n
    result = e[a]
    return result


def de_one_hot(y):
    return np.argmax(y, axis=1)


augmented_data, augmented_results = augment_data(X, y)


c = Counter(de_one_hot(augmented_results))

print(c)


# file_h5 = 'augmented_train.h5'
# f = h5py.File(file_h5, 'w')
# f.create_dataset('X', data=augmented_data)
# f.create_dataset('y', data=augmented_results)
# f.close()
