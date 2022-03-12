# Run

Run with bazelist:

    $ bazelisk build wavegru -c opt --copt=-march=native
    $ ./bazel-bin/wavegru --weight ./weight.pickle --mel ./ft.npy --output audio.wav
