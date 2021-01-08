import numpy as np

a = np.ones((2, 3))

a[...] = a*2
# print(a*2)

b = np.ones((0,))

b[...] = b*2
print(b)
