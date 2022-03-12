## Introduction

We use DeepMind's [sparse matmul library](https://github.com/google/lyra/tree/main/sparse_matmul) to speed-up WaveGRU inference.


## Build
    $ go get github.com/bazelbuild/bazelisk
    $ sudo apt-get install libsndfile1 -y
    $ pip install librosa
    $ bazelisk build wavegru -c opt --copt=-march=native


## Run

    $ ./bazel-bin/wavegru --weight ./weight.pickle --mel ./ft.npy --output audio.wav
