import sys
import time

import numpy as np
from scipy.special import expit
from wavegru_mod import WaveGRU

np.random.seed(42)

print("python is", sys.version_info)


def sigmoid(x):
    return expit(x)


def mygru(fts, embed, m1, b1, m2, b2, m3, b3, o1, o1b, o2, o2b):
    value = 127
    h = np.zeros((512,))
    out = []
    for i in range(fts.shape[0]):
        # embed
        ft = fts[i]
        e = embed[value]
        x = e + ft

        # gru
        t = np.concatenate((x, h), axis=0)
        z = np.matmul(m1, t) + b1
        r = np.matmul(m2, t) + b2
        z = sigmoid(z)
        r = sigmoid(r)
        t = np.concatenate((x, r * h), axis=0)
        hh = np.matmul(m3, t) + b3
        hh = np.tanh(hh)
        h = (1.0 - z) * h + z * hh

        fco1 = np.matmul(o1, h) + o1b
        fco1 = np.maximum(fco1, 0)
        fco2 = np.matmul(o2, fco1) + o2b
        value = np.argmax(fco2)
        out.append(value)

    return out


net = WaveGRU(512, 512)

x = np.linspace(0, 1, 10)
embed = np.random.randn(256, 512).astype(np.float32)
net.load_embed(embed)

m1 = np.random.randn(512, 1024).astype(np.float32)
m2 = np.random.randn(512, 1024).astype(np.float32)
m3 = np.random.randn(512, 1024).astype(np.float32)
b1 = np.random.randn(512).astype(np.float32)
b2 = np.random.randn(512).astype(np.float32)
b3 = np.random.randn(512).astype(np.float32)

mask1 = np.random.randint(0, 100, (512 // 4, 1024 // 4)) >= 90
mask1 = np.tile(mask1[:, None, :, None], (1, 4, 1, 4)).reshape((512, 1024))
mask1 = np.ascontiguousarray(mask1)
m1 = m1 * mask1

mask2 = np.random.randint(0, 100, (512 // 4, 1024 // 4)) >= 90
mask2 = np.tile(mask2[:, None, :, None], (1, 4, 1, 4)).reshape((512, 1024))
mask2 = np.ascontiguousarray(mask2)
m2 = m2 * mask2

mask3 = np.random.randint(0, 100, (512 // 4, 1024 // 4)) >= 90
mask3 = np.tile(mask3[:, None, :, None], (1, 4, 1, 4)).reshape((512, 1024))
mask3 = np.ascontiguousarray(mask3)
m3 = m3 * mask3


o1 = np.random.randn(512, 512).astype(np.float32)
masko1 = np.random.randint(0, 100, (512 // 4, 512 // 4)) >= 90
masko1 = np.tile(masko1[:, None, :, None], (1, 4, 1, 4)).reshape((512, 512))
masko1 = np.ascontiguousarray(masko1)
o1 = o1 * masko1

o1b = np.random.randn(512).astype(np.float32)

o2 = np.random.randn(256, 512).astype(np.float32)
masko2 = np.random.randint(0, 100, (256 // 4, 512 // 4)) >= 90
masko2 = np.tile(masko2[:, None, :, None], (1, 4, 1, 4)).reshape((256, 512))
masko2 = np.ascontiguousarray(masko2)
o2 = o2 * masko2

o2b = np.random.randn(256).astype(np.float32)

net.load_weights(
    m1,
    mask1,
    b1,
    m2,
    mask2,
    b2,
    m3,
    mask3,
    b3,
    o1,
    masko1,
    o1b,
    o2,
    masko2,
    o2b,
)


ft = np.random.randn(100, 512).astype(np.float32)

kk = mygru(
    ft,
    embed,
    m1,
    b1,
    m2,
    b2,
    m3,
    b3,
    o1,
    o1b,
    o2,
    o2b,
)
print(kk)


signal = net.inference(ft, 1e-5)
print(signal)


ft = np.random.randn(16_000, 512).astype(np.float32)
start = time.perf_counter()
signal = net.inference(ft, 1.0)
end = time.perf_counter()
print(end - start)
