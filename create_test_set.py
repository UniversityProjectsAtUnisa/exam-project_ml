import h5py

file_h5 = './data/train.h5'
f = h5py.File(file_h5, 'r')
X = f['X'][...]
y = f['y'][...]
f.close()
