from visualize import visualize_element
import h5py

file_h5 = './data/augmented_train.h5'
f = h5py.File(file_h5, 'r')
X = f['X'][:100]
y = f['y'][:100]
f.close()


visualize_element(X[0], y[0])