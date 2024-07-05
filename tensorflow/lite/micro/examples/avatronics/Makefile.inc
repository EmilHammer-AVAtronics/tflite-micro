AVATRONICS_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/avatronics_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/airoha_nn_model.cc

# CONFIG_FILE := \
# $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/config.h

# FILES_TO_BE_TOUCHED := \
# $(TENSORFLOW_ROOT)tensorflow/lite/micro/micro_interpreter_graph.cc \
# $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/conv.cc \
# $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/xtensa/conv_hifi.cc \
# $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/conv.cc

AVATRONICS_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/avatronics_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/airoha_nn_model.cc

AVATRONICS_HDRS :=

TFLITE_DIR := $(TENSORFLOW_ROOT)../../shared_ubuntu/AVAHEEAR/AVA_DPCRN/pretrained_weights/DPCRN_S/airoha_models/
TFLITE_FILES := $(wildcard $(TFLITE_DIR)*.tflite)
TFLITE_BASENAMES := $(patsubst $(TFLITE_DIR)%.tflite,%,$(TFLITE_FILES))

$(info Files in $(TFLITE_DIR): $(TFLITE_BASENAMES))

# Generates .html visualizations of .tflite files
$(foreach file,$(TFLITE_BASENAMES), $(shell python3 $(TENSORFLOW_ROOT)tflite_visualize.py $(TFLITE_DIR)$(file).tflite $(TFLITE_DIR)$(file)_visualization.html))

AVATRONICS_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/models/DPCRN_m71_quant_full_int_InOuts_float32_13_6_2024.tflite
# $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/models/DPCRN_m71_quant_full_int_InOuts_float32.tflite
# $(foreach file, $(TFLITE_BASENAMES), $(TFLITE_DIR)$(file).tflite)

AVATRONICS_GENERATED_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/models/DPCRN_m71_quant_full_int_InOuts_float32_13_6_2024_model_data.cc
# $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/models/DPCRN_m71_quant_full_int_InOuts_float32_model_data.cc
# $(foreach file, $(TFLITE_BASENAMES), $(GENERATED_SRCS_DIR)$(TFLITE_DIR)$(file).cc)

AVATRONICS_GENERATED_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/models/DPCRN_m71_quant_full_int_InOuts_float32_13_6_2024_model_data.h
# $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/models/DPCRN_m71_quant_full_int_InOuts_float32_model_data.h
# $(foreach file, $(TFLITE_BASENAMES), $(GENERATED_SRCS_DIR)$(TFLITE_DIR)$(file).h)

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