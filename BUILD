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
        "gru.cc",
    ],
)

pybind_extension(
    name = "my_pb_mod",  # This name is not actually created!
    srcs = ["my_pb_mod.cc"],
)

py_library(
    name = "my_pb_mod",
    data = [":my_pb_mod.so"],
)

py_binary(
    name = "example",
    srcs = ["example.py"],
    deps = [
        ":my_pb_mod"
    ],
)


cc_binary(
    name = "gru",
    srcs = [
        "gru.cc",
    ],
    deps = [
        "//sparse_matmul",
    ],
)

