import time
import os
import numpy as np
import sys
from wavernn_mod import WaveRNN

print("python is", sys.version_info)
print("imported, about to call", file=sys.stderr)

net = WaveRNN(512, 512)

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

mask2 = np.random.randint(0, 100, (512 // 4, 1024 // 4)) >= 90
mask2 = np.tile(mask2[:, None, :, None], (1, 4, 1, 4)).reshape((512, 1024))
mask2 = np.ascontiguousarray(mask2)

mask3 = np.random.randint(0, 100, (512 // 4, 1024 // 4)) >= 90
mask3 = np.tile(mask3[:, None, :, None], (1, 4, 1, 4)).reshape((512, 1024))
mask3 = np.ascontiguousarray(mask3)


o1 = np.random.randn(512, 512).astype(np.float32)
masko1 = np.random.randint(0, 100, (512 // 4, 512 // 4)) >= 90
masko1 = np.tile(masko1[:, None, :, None], (1, 4, 1, 4)).reshape((512, 512))
masko1 = np.ascontiguousarray(masko1)
o1b = np.random.randn(512).astype(np.float32)

o2 = np.random.randn(256, 512).astype(np.float32)
masko2 = np.random.randint(0, 100, (256 // 4, 512 // 4)) >= 90
masko2 = np.tile(masko2[:, None, :, None], (1, 4, 1, 4)).reshape((256, 512))
masko2 = np.ascontiguousarray(masko2)
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


ft = np.random.randn(16000, 512).astype(np.float32)

start = time.perf_counter()
signal = net.inference(ft)
end = time.perf_counter()
print(end - start)
