/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <math.h>
#include <string>
#include "meta_test_fct.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/examples/meta/models/meta_float_model_data.h" // Meta example model
#include "tensorflow/lite/micro/examples/meta/models/meta_int8_model_data.h"  // Meta example model
#include "tensorflow/lite/micro/examples/meta/models/micro_ex/micro_speech_quantized_model_data.cc" // micro_speech_quantized_model_data example model

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"

#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/examples/meta/input749.h"
#include "tensorflow/lite/micro/examples/meta/inputDPCRN.h"
#include "tensorflow/lite/micro/examples/meta/output749.h"

/*============================== MODELS =======================================*/
// #include "tensorflow/lite/micro/examples/meta/models/test_models/1242/DPCRN-m32-quant-full-int-InOuts-float32_flatbuffer.h"
// #include "tensorflow/lite/micro/examples/meta/models/test_models/1242/DPCRN-m32-quant-full-int_flatbuffer.h"
// #include "tensorflow/lite/micro/examples/meta/models/test_models/1311/DPCRN-m32-quant-float16_flatbuffer.h"
// #include "tensorflow/lite/micro/examples/meta/models/test_models/1311/DPCRN-m32-quant-full-int_flatbuffer.h"
// #include "tensorflow/lite/micro/examples/meta/models/test_models/1311/DPCRN-m32_flatbuffer.h"
// #include "tensorflow/lite/micro/examples/meta/models/test_models/1311/DPCRN-m32-quant_flatbuffer.h"
// #include "tensorflow/lite/micro/examples/meta/models/test_models/1311/DPCRN-m32-quant-full-int-float32interace_flatbuffer.h"

#define PROFILE
#define PROF_ALLOCATE
#include "xt_profiler.h"

// Namespace
namespace {
using MetaOpResolver = tflite::MicroMutableOpResolver<29>;

TfLiteStatus RegisterOps(MetaOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPad());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddUnpack());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLogistic());
  TF_LITE_ENSURE_STATUS(op_resolver.AddTanh());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSub());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPack());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMean());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSquaredDifference());  
  TF_LITE_ENSURE_STATUS(op_resolver.AddRsqrt());
  TF_LITE_ENSURE_STATUS(op_resolver.AddStridedSlice());
  TF_LITE_ENSURE_STATUS(op_resolver.AddWhile());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConcatenation());
  TF_LITE_ENSURE_STATUS(op_resolver.AddTransposeConv()); 
  // TF_LITE_ENSURE_STATUS(op_resolver.AddPOW()); // op code doesn't exits !!! Experimental (i.e doesn't work) 
  TF_LITE_ENSURE_STATUS(op_resolver.AddExp());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLog());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLess());
  TF_LITE_ENSURE_STATUS(op_resolver.AddGather()); 
  TF_LITE_ENSURE_STATUS(op_resolver.AddSlice());
  TF_LITE_ENSURE_STATUS(op_resolver.AddCast());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDequantize());
  TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
  // Needed for the working and defualt model g_meta_float_model_data
  TF_LITE_ENSURE_STATUS(op_resolver.AddMaxPool2D());
   // Needed for the working and defualt model g_micro_speech_quantized_model_data
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  return kTfLiteOk;
}

tflite::MicroInterpreter* interpreter = nullptr;
constexpr int kTensorArenaSize = 40 * 1024;
// constexpr int kTensorArenaSize = 421376;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = ::tflite::GetModel(g_meta_float_model_data); 
// const tflite::Model* model = ::tflite::GetModel(DPCRN_m32_quant_tflite);
// const tflite::Model* model = ::tflite::GetModel(DPCRN_m32_quant_full_int_tflite);
// const tflite::Model* model = ::tflite::GetModel(DPCRN_m32_quant_float16_tflite);
// const tflite::Model* model = ::tflite::GetModel(DPCRN_m32_quant_full_int_float32interace_tflite);
// const tflite::Model* model = ::tflite::GetModel(__DPCRN_m32_quant_full_int_InOuts_float32_tflite);
// const tflite::Model* model = ::tflite::GetModel(DPCRN_m32_quant_full_int_InOuts_float32_tflite);
// const tflite::Model* model = ::tflite::GetModel(__DPCRN_m21_tflite);
// const tflite::Model* model = ::tflite::GetModel(g_micro_speech_quantized_model_data);
// const tflite::Model* model = ::tflite::GetModel(DPCRN_m32_quant_full_int_tflite);
// const tflite::Model* model = ::tflite::GetModel(DPCRN_m32_quant_full_int_InOuts_float32_tflite);

}  // namespace


TfLiteStatus LoadFloatModelAndPerformInference(uint8_t* y_pred0_out1, uint8_t* y_pred1_out1) {
  tflite::InitializeTarget();

  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);
  MetaOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());
  MicroPrintf("Bytes required: %u", interpreter.arena_used_bytes());

  constexpr int kNumTestValues = TVX*TVY;
  MicroPrintf("Start: interpreter.input(0)->data.f[i]");
  for (int i = 0; i < kNumTestValues; ++i) {
    interpreter.input(0)->data.f[i] = qCorr_24sum[i];
  }

  MicroPrintf("Start: interpreter.Invoke()");
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());

  MicroPrintf("Start: interpreter.output(0)->data.f[0]");
  *y_pred0_out1 = uint8_t(interpreter.output(0)->data.f[0]);
  *y_pred1_out1 = uint8_t(interpreter.output(0)->data.f[1]);

  free(tensor_arena);

  return kTfLiteOk;
}

void nn_setup() {
  tflite::InitializeTarget();
  // model = tflite::GetModel(DPCRN_m32_quant_full_int_float32interace_tflite); // input->type != kTfLiteFloat32 (INT8 != FLOAT32)

  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION); 

  MetaOpResolver op_resolver;
  RegisterOps(op_resolver);

  static tflite::MicroMutableOpResolver<27> micro_op_resolver;
  micro_op_resolver.AddMul();
  micro_op_resolver.AddAdd();


  static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }
  return;
}

void nn_inference(uint8_t* y_pred0_out1, uint8_t* y_pred1_out1) {
  char profiler_name[MAX_PROFILER_NAME_LENGTH];
  char profiler_params[MAX_PROFILER_PARAMS_LENGTH];
  int num_ops=0;
  constexpr int kNumTestValues = TVX*TVY;

  for (int i = 0; i < kNumTestValues; ++i) {
    interpreter->input(0)->data.f[i] = qCorr_24sum[i];
  }

  XTPWR_PROFILER_OPEN(0, profiler_name, profiler_params, num_ops, "OPs/cyc", 1);
  XTPWR_PROFILER_UPDATE(0);
  XTPWR_PROFILER_START(0);
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }
  XTPWR_PROFILER_STOP(0);
  XTPWR_PROFILER_UPDATE(0);
  XTPWR_PROFILER_PRINT(0);

  // Process the inference results.
  *y_pred0_out1 = uint8_t(interpreter->output(0)->data.f[0]);
  *y_pred1_out1 = uint8_t(interpreter->output(0)->data.f[1]);
  return;
} 

int wrapper_nn_setup(){
  // nn_setup();
  return 0;
}

int wrapper_nn_inference(uint8_t* y_pred0_out1, uint8_t* y_pred1_out1){
  nn_inference(y_pred0_out1, y_pred1_out1);
  return 0;
}

int cpp_meta_test_fct() {

  uint8_t a1 = 0;
  uint8_t a2 = 0;

  uint8_t* y_pred0_out1 = &a1; 
  uint8_t* y_pred1_out1 = &a2;

  MicroPrintf("\nStart: LoadFloatModelAndPerformInference(y_pred0_out1, y_pred1_out1)");
  TF_LITE_ENSURE_STATUS(LoadFloatModelAndPerformInference(y_pred0_out1, y_pred1_out1));
  MicroPrintf("\n~~~ LoadFloatModelAndPerformInference ~~~");
  MicroPrintf("pred0 %d", *y_pred0_out1);
  MicroPrintf("pred1 %d", *y_pred1_out1);

  // MicroPrintf("Start of: nn_setup()\n");
  // nn_setup();
  // MicroPrintf("Start of: nn_inference()\n");
  // nn_inference(y_pred0_out1, y_pred1_out1);


  // MicroPrintf("~~~ PerformAll ~~~");
  // MicroPrintf("pred0 %d\n", *y_pred0_out1);
  // MicroPrintf("pred1 %d\n", *y_pred1_out1);
  
  MicroPrintf("~~~ALL TESTS PASSED~~~\n");
  return 0;
}





