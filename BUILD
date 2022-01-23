# [internal] load cc_fuzz_target.bzl
# [internal] load cc_proto_library.bzl
# [internal] load android_cc_test:def.bzl

package(default_visibility = [":__subpackages__"])

licenses(["notice"])

# To run all cc_tests in this directory:
# bazel test //:all

# [internal] Command to run dsp_util_android_test.

# [internal] Command to run lyra_integration_android_test.

exports_files(
    srcs = [
        "wavegru.cc",
    ],
)


cc_binary(
    name = "wavegru",
    srcs = [
        "wavegru.cc",
    ],
    deps = [
        "//sparse_matmul",
    ],
)

