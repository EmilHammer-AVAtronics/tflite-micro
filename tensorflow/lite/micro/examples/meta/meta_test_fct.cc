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

#include "meta_test_fct.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/examples/meta/models/meta_float_model_data.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"

#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/examples/meta/input749.h"

#define PROFILE
#define PROF_ALLOCATE
#include "xt_profiler.h"

namespace {
tflite::MicroInterpreter* global_interpreter = nullptr;
constexpr int kTensorArenaSize = 4000;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
const tflite::Model* model = ::tflite::GetModel(g_meta_float_model_data); // <-- Original
}

int meta_setup() {
  tflite::InitializeTarget();

  model = tflite::GetModel(g_meta_float_model_data); 
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION); 

  static tflite::MicroMutableOpResolver<4> micro_op_resolver;
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddMaxPool2D();

  static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  global_interpreter = &static_interpreter;

  if (global_interpreter == nullptr) return 1;

  TfLiteStatus allocate_status = global_interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) return 2;

  return 0;
}

int meta_inference(uint8_t* y_pred0_out1, uint8_t* y_pred1_out1) {
  constexpr int kNumTestValues = TVX*TVY;
  for (int i = 0; i < kNumTestValues; ++i) {
    global_interpreter->input(0)->data.f[i] = qCorr_24sum[i];
  }
  if (kTfLiteOk != global_interpreter->Invoke()) return 1;

  *y_pred0_out1 = uint8_t(global_interpreter->output(0)->data.f[0]);
  *y_pred1_out1 = uint8_t(global_interpreter->output(0)->data.f[1]);
  return 0;
} 

int wrapper_meta_setup(){
  int ret = meta_setup();
  return ret;
}

int wrapper_meta_inference(uint8_t* y_pred0_out1, uint8_t* y_pred1_out1){
  int ret = meta_inference(y_pred0_out1, y_pred1_out1);
  return ret;
}

/* TFLM test of the functions */
int cpp_meta_test_fct() {
  uint8_t a1 = 0;
  uint8_t a2 = 0;
  uint8_t* y_pred0_out1 = &a1; 
  uint8_t* y_pred1_out1 = &a2;

  MicroPrintf("\nStart:\n");
  wrapper_meta_setup();
  wrapper_meta_inference(y_pred0_out1,y_pred1_out1);

  
  MicroPrintf("\npred0 %d", *y_pred0_out1);
  MicroPrintf("\npred1 %d\n\n", *y_pred1_out1);

  if((*y_pred0_out1 == 18) && (*y_pred1_out1 == 81)) {
    MicroPrintf("~~~ALL TESTS PASSED~~~\n");
  } else {
    MicroPrintf("The predications does not match the expected values!!! \n");
    MicroPrintf("*y_pred0_out1 != 18, y_pred0_out1: %d\n", *y_pred0_out1);
    MicroPrintf("*y_pred1_out1 != 81, y_pred0_out1: %d\n", *y_pred1_out1);
  }
  return 0;
}