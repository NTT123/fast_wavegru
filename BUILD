# [internal] load cc_fuzz_target.bzl
# [internal] load cc_proto_library.bzl
# [internal] load android_cc_test:def.bzl

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = [":__subpackages__"])

licenses(["notice"])

# To run all cc_tests in this directory:
# bazel test //:all

# [internal] Command to run dsp_util_android_test.

# [internal] Command to run lyra_integration_android_test.

exports_files(
    srcs = [
        "gru.h",
    ],
)

pybind_extension(
    name = "wavernn_mod",  # This name is not actually created!
    srcs = ["wavernn_mod.cc"],
    deps = [
        ":gru"
    ],
)

py_library(
    name = "wavernn_mod",
    data = [":wavernn_mod.so"],
)

py_binary(
    name = "speak",
    srcs = ["speak.py"],
    deps = [
        ":wavernn_mod"
    ],
)


cc_library(
    name = "gru",
    srcs = [
        "gru.h",
    ],
    deps = [
        "//sparse_matmul",
    ],
)




pybind_extension(
    name = "m1_mod",  # This name is not actually created!
    srcs = ["m1_mod.cc"],
    deps = [
        "//sparse_matmul"
    ],
)

py_library(
    name = "m1_mod",
    data = [":m1_mod.so"],
)

py_binary(
    name = "m1",
    srcs = ["m1.py"],
    deps = [
        ":m1_mod"
    ],
)

