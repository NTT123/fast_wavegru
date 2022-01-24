import sys

import numpy as np
from m1_mod import M1

print("python is", sys.version_info)
print("imported, about to call", file=sys.stderr)

np.random.seed(42)


def mygru(ft, a, b):
    z = np.matmul(a, ft) + b
    return z


m1 = np.random.randn(4, 4).astype(np.float32)
b1 = np.random.randn(4).astype(np.float32)
mask1 = np.random.randint(0, 100, (4 // 4, 4 // 4)) >= 95
mask1 = np.tile(mask1[:, None, :, None], (1, 4, 1, 4)).reshape((4, 4))
mask1 = np.ascontiguousarray(mask1.astype(np.int16))
m1 = m1 * mask1
x = np.random.randn(4).astype(np.float32)
k = mygru(x, m1, b1)
print(k)

net = M1(4, 4)
net.load_weights(m1, mask1, b1)
net.forward(x)
