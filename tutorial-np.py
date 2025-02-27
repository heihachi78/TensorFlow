import numpy as np
import time

np_random_array1 = np.random.rand(6400, 6400)
np_random_array2 = np.random.rand(6400, 6400)

s = time.perf_counter()
np_mul = np.dot(np_random_array1, np_random_array2)
e = time.perf_counter()
print(e-s)