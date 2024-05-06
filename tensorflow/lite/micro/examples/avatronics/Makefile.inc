AVATRONICS_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/avatronics_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/Airoha_DPCRN_m34_quant_full_int.cc

AVATRONICS_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/avatronics_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/Airoha_DPCRN_m34_quant_full_int.cc

AVATRONICS_HDRS :=

AVATRONICS_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/models/DPCRN_m34_quant_full_int.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/models/DPCRN_m34.tflite

AVATRONICS_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/models/DPCRN_m34_quant_full_int.cc \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/models/DPCRN_m34.cc

AVATRONICS_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/models/DPCRN_m34_quant_full_int.h \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/models/DPCRN_m34.h

# Tests loading and running the sine model.
$(eval $(call microlite_test,avatronics_test,\
$(AVATRONICS_TEST_SRCS),,$(AVATRONICS_GENERATOR_INPUTS)))

# Builds a standalone binary.
$(eval $(call microlite_test,avatronics,\
$(AVATRONICS_SRCS),,$(AVATRONICS_GENERATOR_INPUTS)))

# Add sources and headers generated from $(AVATRONICS_GENERATOR_INPUTS).
AVATRONICS_SRCS += $(AVATRONICS_GENERATED_SRCS)
AVATRONICS_HDRS += $(AVATRONICS_GENERATED_HDRS)

list_avatronics_example_sources:
	@echo $(AVATRONICS_SRCS)

list_avatronics_example_headers:
	@echo $(AVATRONICS_HDRS)

