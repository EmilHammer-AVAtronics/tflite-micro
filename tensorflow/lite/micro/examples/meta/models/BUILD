load("//tensorflow/lite/micro:build_def.bzl", "generate_cc_arrays")

package(
    default_visibility = ["//visibility:public"],
)

exports_files(
    srcs = [
        "meta_float.tflite",
#        "meta_int8.tflite",
    ],
    visibility = ["//tensorflow/lite/micro/examples/meta:__subpackages__"],
)

generate_cc_arrays(
    name = "generated_meta_float_model_cc",
    src = "meta_float.tflite",
    out = "meta_float_model_data.cc",
)

generate_cc_arrays(
    name = "generated_meta_float_model_hdr",
    src = "meta_float.tflite",
    out = "meta_float_model_data.h",
)

generate_cc_arrays(
    name = "generated_meta_int8_model_cc",
    src = "meta_int8.tflite",
    out = "meta_int8_model_data.cc",
)

generate_cc_arrays(
    name = "generated_meta_int8_model_hdr",
    src = "meta_int8.tflite",
    out = "meta_int8_model_data.h",
)
