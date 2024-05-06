META_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/meta/meta_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/meta/meta_test_fct.cc

META_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/meta/meta_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/meta/meta_test_fct.cc

META_HDRS :=

META_GENERATOR_INPUTS := \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/meta/models/meta_float.tflite \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/meta/models/meta_int8.tflite
# $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/meta/models/DPCRN-m8-quant-float16.tflite

META_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/meta/models/meta_float_model_data.cc \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/meta/models/meta_int8_model_data.cc

META_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/meta/models/meta_float_model_data.h \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/meta/models/meta_int8_model_data.h \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/meta/models/DPCRN-m8-quant-float16_flatbuffer.h

# Tests loading and running the sine model.
$(eval $(call microlite_test,meta_test,\
$(META_TEST_SRCS),,$(META_GENERATOR_INPUTS)))

# Builds a standalone binary.
$(eval $(call microlite_test,meta,\
$(META_SRCS),,$(META_GENERATOR_INPUTS)))

# Add sources and headers generated from $(META_GENERATOR_INPUTS).
META_SRCS += $(META_GENERATED_SRCS)
META_HDRS += $(META_GENERATED_HDRS)

list_meta_example_sources:
	@echo $(META_SRCS)

list_meta_example_headers:
	@echo $(META_HDRS)
