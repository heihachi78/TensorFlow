import numpy as np
import time
import sys

s = time.perf_counter()
np_random_array1 = np.random.rand(8192, 8192 * 4)
print(sys.getsizeof(np_random_array1) / 1024 / 1024 / 1024, 'GB')
e = time.perf_counter()
print(e-s, ' sec')
np_random_array2 = np.random.rand(8192, 8192 * 4)
print(sys.getsizeof(np_random_array2) / 1024 / 1024 / 1024, 'GB')
e = time.perf_counter()
print(e-s, ' sec')
#np_mul = np.dot(np_random_array1, np_random_array2)
e = time.perf_counter()
print(e-s, ' sec')