import pickle
import time
from argparse import ArgumentParser
from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile
from wavegru_mod import WaveGRU

parser = ArgumentParser()
parser.add_argument("--weight", type=Path, help="Path to weight file", required=True)
parser.add_argument("--mel", type=Path, help="Path to mel file", required=True)
parser.add_argument(
    "--output", type=Path, help="Path to output wav file", required=True
)
args = parser.parse_args()

with open(args.weight, "rb") as f:
    data = pickle.load(f)

embed = data["embed_weight"]


embed_dim = embed.shape[1]
rnn_dim = data["gru_xh_h_bias"].shape[0]
input_dim = data["gru_xh_zr_weight"].shape[1] - rnn_dim
net = WaveGRU(input_dim, embed_dim, rnn_dim)
net.load_embed(embed)
dim = embed_dim + input_dim + rnn_dim
z, r = np.split(data["gru_xh_zr_weight"].T, 2, axis=0)
h = data["gru_xh_h_weight"].T
z = np.ascontiguousarray(z)
r = np.ascontiguousarray(r)
h = np.ascontiguousarray(h)

b1, b2 = np.split(data["gru_xh_zr_bias"], 2)
b3 = data["gru_xh_h_bias"]
m1, m2, m3 = z, r, h

mask_z, mask_r = np.split(data["gru_xh_zr_mask"].T, 2, axis=0)
mask_h = data["gru_xh_h_mask"].T
mask_z = np.ascontiguousarray(mask_z)
mask_r = np.ascontiguousarray(mask_r)
mask_h = np.ascontiguousarray(mask_h)

mask1, mask2, mask3 = mask_z, mask_r, mask_h

o1 = np.ascontiguousarray(data["o1_weight"].T)
masko1 = np.ascontiguousarray(data["o1_mask"].T)
o1b = data["o1_bias"]

o2 = np.ascontiguousarray(data["o2_weight"].T)
masko2 = np.ascontiguousarray(data["o2_mask"].T)
o2b = data["o2_bias"]


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


ft = np.load(args.mel)
if len(ft.shape) == 3:
    ft = ft[0]
start = time.perf_counter()
signal = net.inference(ft, 1.0)
end = time.perf_counter()
print("Duration:", end - start)

signal = np.array(signal)
wav = librosa.mu_expand(signal - 127, mu=255)
wav = librosa.effects.deemphasis(wav, coef=0.86)
wav = wav * 2.0
wav = wav / max(1.0, np.max(np.abs(wav)))
wav = wav * 2 ** 15
wav = np.clip(wav, a_min=-(2 ** 15), a_max=(2 ** 15) - 1)
wav = wav.astype(np.int16)

wavfile.write(str(args.output), 24_000, wav)
print(f"Saved output to '{args.output}'")
