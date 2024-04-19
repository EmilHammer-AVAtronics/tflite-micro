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

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/examples/meta/models/meta_float_model_data.h"
#include "tensorflow/lite/micro/examples/meta/models/meta_int8_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/examples/meta/input749.h"
#include "tensorflow/lite/micro/examples/meta/output749.h"

namespace {
using MetaOpResolver = tflite::MicroMutableOpResolver<20>;

TfLiteStatus RegisterOps(MetaOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMaxPool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDequantize());
  return kTfLiteOk;
}
}  // namespace

//TfLiteStatus ProfileMemoryAndLatency() {
//  tflite::MicroProfiler profiler;
//  MetaOpResolver op_resolver;
//  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));
//
//  // Arena size just a round number. The exact arena usage can be determined
//  // using the RecordingMicroInterpreter.
//  constexpr int kTensorArenaSize = 5000;
//  uint8_t tensor_arena[kTensorArenaSize];
//  constexpr int kNumResourceVariables = 24;
//
//  tflite::RecordingMicroAllocator* allocator(
//      tflite::RecordingMicroAllocator::Create(tensor_arena, kTensorArenaSize));
//  tflite::RecordingMicroInterpreter interpreter(
//      tflite::GetModel(g_meta_float_model_data), op_resolver, allocator,
//      tflite::MicroResourceVariables::Create(allocator, kNumResourceVariables),
//      &profiler);
//
//  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());
//  TFLITE_CHECK_EQ(interpreter.inputs_size(), 1);
//  interpreter.input(0)->data.int8[0] = 1;
//  TF_LITE_ENSURE_STATUS(interpreter.Invoke());
//
//  MicroPrintf("");  // Print an empty new line
//  profiler.LogTicksPerTagCsv();
//
//  MicroPrintf("");  // Print an empty new line
//  interpreter.GetMicroAllocator().PrintAllocations();
//  return kTfLiteOk;
//}

TfLiteStatus LoadFloatModelAndPerformInference() {
  const tflite::Model* model =
      ::tflite::GetModel(g_meta_float_model_data);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  MetaOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 5000;
  //uint8_t tensor_arena[kTensorArenaSize];

  uint8_t* tensor_arena;
  tensor_arena = (uint8_t*) malloc( (size_t) kTensorArenaSize * sizeof(uint8_t));
  
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  // Check if the predicted output is within a small range of the
  // expected output
  //  float epsilon = 0.05f;
  constexpr int kNumTestValues = TVX*TVY;
  //  float golden_inputs[kNumTestValues] = {0.f, 1.f, 3.f, 5.f};

  for (int i = 0; i < kNumTestValues; ++i) {
    //interpreter.input(0)->data.f[0] = golden_inputs[i];
    interpreter.input(0)->data.f[i] = qCorr_24sum[i];
  }
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());
  double y_pred0 = double(interpreter.output(0)->data.f[0]);
  double y_pred1 = double(interpreter.output(0)->data.f[1]);
  printf("pred0 %f\n", y_pred0);
  printf("pred1 %f\n", y_pred1);

  printf("Zero inputs:\n");
  for (int i = 0; i < kNumTestValues; ++i) {
    //interpreter.input(0)->data.f[0] = golden_inputs[i];
    interpreter.input(0)->data.f[i] = 0.0;
  }
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());
  y_pred0 = double(interpreter.output(0)->data.f[0]);
  y_pred1 = double(interpreter.output(0)->data.f[1]);
  printf("pred0 %f\n", y_pred0);
  printf("pred1 %f\n", y_pred1);

  printf("y*TVX+x:\n");    
  int i=0;
  for (int x = 0; x < TVX; x++)
    for(int y = 0; y < TVY; y++)
      {
	interpreter.input(0)->data.f[i++] = qCorr_24sum[y*TVX+x];
      }
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());
  y_pred0 = double(interpreter.output(0)->data.f[0]);
  y_pred1 = double(interpreter.output(0)->data.f[1]);
  printf("pred0 %f\n", y_pred0);
  printf("pred1 %f\n", y_pred1);

  printf("x*TVY+y:\n");    
  i=0;
  for (int x = 0; x < TVX; x++)
    for(int y = 0; y < TVY; y++)
      {
	interpreter.input(0)->data.f[i++] = qCorr_24sum[x*TVY+y];
      }
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());
  y_pred0 = double(interpreter.output(0)->data.f[0]);
  y_pred1 = double(interpreter.output(0)->data.f[1]);
  printf("pred0 %f\n", y_pred0);
  printf("pred1 %f\n", y_pred1);

  //    printf("output 0: %lf,  1: %lf \n",  y_pred0,  y_pred1); 
  //    TFLITE_CHECK_LE(abs(sin(golden_inputs[i]) - y_pred), epsilon);

 
  free(tensor_arena); 
  return kTfLiteOk;
}


TfLiteStatus LoadQuantModelAndPerformInference() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      ::tflite::GetModel(g_meta_int8_model_data);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  MetaOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSizeQ = 10000;
  uint8_t tensor_arena[kTensorArenaSizeQ];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSizeQ);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  constexpr int kNumTestValuesQ = TVX*TVY;
  //  float golden_inputs[kNumTestValues] = {0.f, 1.f, 3.f, 5.f};
    
  TfLiteTensor* input = interpreter.input(0);
  TFLITE_CHECK_NE(input, nullptr);

  TfLiteTensor* output = interpreter.output(0);
  TFLITE_CHECK_NE(output, nullptr);

  double output_scale = (double) output->params.scale;
  int output_zero_point = output->params.zero_point;
  int input_scale = input->params.scale;
  int input_zero_point = input->params.zero_point;

  // Check if the predicted output is within a small range of the
  // expected output
  //float epsilon = 0.05;

  //constexpr int kNumTestValues = 4;
  //float golden_inputs_float[kNumTestValues] = {0.77, 1.57, 2.3, 3.14};

  // The int8 values are calculated using the following formula
  // (golden_inputs_float[i] / input->params.scale + input->params.scale)
  //int8_t golden_inputs_int8[kNumTestValues] = {-96, -63, -34, 0};

  //for (int i = 0; i < kNumTestValues; ++i) {
  //  input->data.int8[0] = golden_inputs_int8[i];
  //  TF_LITE_ENSURE_STATUS(interpreter.Invoke());
  //  float y_pred = (output->data.int8[0] - output_zero_point) * output_scale;
  //  TFLITE_CHECK_LE(abs(sin(golden_inputs_float[i]) - y_pred), epsilon);
  //}

  printf("\nInt8 model\n");
  printf("output_scale %f\n", output_scale);
  printf("output_zero_point %d\n", output_zero_point);
  printf("input_scale %d\n", input_scale);
  printf("input_zero_point %d\n", input_zero_point);
  
//  for (int i = 0; i < kNumTestValuesQ; ++i) {
//    //interpreter.input(0)->data.f[0] = golden_inputs[i];
//    interpreter.input(0)->data.int8[i] =(int) qCorr_24sum[i];
//  }
//
//  TF_LITE_ENSURE_STATUS(interpreter.Invoke());
//  int y_pred0 = int(interpreter.output(0)->data.int8[0]);
//  int y_pred1 = int(interpreter.output(0)->data.int8[1]);
  int y_pred0;
  int y_pred1;
  double y_pred0f;
  double y_pred1f;
//  printf("int8 pred0 %d\n", y_pred0);
//  printf("int8 pred1 %d\n", y_pred1);


  
  printf("Zero inputs:\n");
  for (int i = 0; i < kNumTestValuesQ; ++i) {
    interpreter.input(0)->data.f[0] = 0.0;
    //interpreter.input(0)->data.f[i] = qCorr_24sum[i];
    interpreter.input(0)->data.int8[i] = 0;
    //input->data.int8[i]=0;
  }
  printf("Input written\n");
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());
  y_pred0 = interpreter.output(0)->data.int8[0];
  y_pred1 = interpreter.output(0)->data.int8[1];
  printf("int8 pred0 %d\n", y_pred0);
  printf("int8 pred1 %d\n", y_pred1);

  y_pred0f = double(interpreter.output(0)->data.f[0]);
  y_pred1f = double(interpreter.output(0)->data.f[1]);
  printf("pred0 %f\n", y_pred0f);
  printf("pred1 %f\n", y_pred1f);

  printf("inputs vector:\n");
  for (int i = 0; i < kNumTestValuesQ; ++i) {
    //interpreter.input(0)->data.f[0] = 0.0;
    interpreter.input(0)->data.f[i] = qCorr_24sum[i];
    //interpreter.input(0)->data.int8[i] = 0;
    //input->data.int8[i]=0;
  }
  printf("Input written\n");
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());
  y_pred0 = interpreter.output(0)->data.int8[0];
  y_pred1 = interpreter.output(0)->data.int8[1];
  printf("int8 pred0 %d\n", y_pred0);
  printf("int8 pred1 %d\n", y_pred1);

  y_pred0f = double(interpreter.output(0)->data.f[0]);
  y_pred1f = double(interpreter.output(0)->data.f[1]);
  printf("pred0 %f\n", y_pred0f);
  printf("pred1 %f\n", y_pred1f);

  return kTfLiteOk;
}


int meta_test_fct() {
  tflite::InitializeTarget();
  //  TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency());
  TF_LITE_ENSURE_STATUS(LoadFloatModelAndPerformInference());
  TF_LITE_ENSURE_STATUS(LoadQuantModelAndPerformInference());
  MicroPrintf("~~~ALL TESTS PASSED~~~\n");
  return kTfLiteOk;
}
