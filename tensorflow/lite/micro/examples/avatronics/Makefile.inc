AVATRONICS_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/avatronics_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/Airoha_DPCRN_m34_quant_full_int.cc

AVATRONICS_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/avatronics_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/Airoha_DPCRN_m34_quant_full_int.cc

AVATRONICS_HDRS :=

TFLITE_DIR := $(TENSORFLOW_ROOT)../../shared_ubuntu/AVAHEEAR/AVA_DPCRN/pretrained_weights/DPCRN_S/airoha_models/
TFLITE_FILES := $(wildcard $(TFLITE_DIR)*.tflite)
TFLITE_BASENAMES := $(patsubst $(TFLITE_DIR)%.tflite,%,$(TFLITE_FILES))

$(info Files in $(TFLITE_DIR): $(TFLITE_BASENAMES))


# Generates .html visualizations of .tflite files
$(foreach file,$(TFLITE_BASENAMES), $(shell python3 $(TENSORFLOW_ROOT)tflite_visualize.py $(TFLITE_DIR)$(file).tflite $(TFLITE_DIR)$(file)_visualization.html))

AVATRONICS_GENERATOR_INPUTS := \
    $(foreach file, $(TFLITE_BASENAMES), $(TFLITE_DIR)$(file).tflite) 
    # $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/models/DPCRN_m34_quant_full_int.tflite \
    # $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/models/DPCRN_m34.tflite

AVATRONICS_GENERATED_SRCS := \
    $(foreach file, $(TFLITE_BASENAMES), $(GENERATED_SRCS_DIR)$(TFLITE_DIR)$(file).cc) 
    # $(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/models/DPCRN_m34_quant_full_int.cc \
    # $(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/models/DPCRN_m34.cc

AVATRONICS_GENERATED_HDRS := \
    $(foreach file, $(TFLITE_BASENAMES), $(GENERATED_SRCS_DIR)$(TFLITE_DIR)$(file).h) 
    # $(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/models/DPCRN_m34_quant_full_int.h \
    # $(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/avatronics/models/DPCRN_m34.h


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