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

#include "Airoha_DPCRN_m34_quant_full_int.h"
#include "tensorflow/lite/core/c/common.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"

#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/examples/meta/inputDPCRN.h"

/*============================== MODELS =======================================*/
const char* model_name = "g_DPCRN_m34_quant_full_int_model_data";  

/*============ DPCRN-m34 COMMIT: April 30th, 2024 10:30 AM ======================*/
#include "tensorflow/lite/micro/examples/avatronics/models/DPCRN_m34_quant_full_int_model_data.h"
#include "tensorflow/lite/micro/examples/meta/models/meta_float_model_data.h"
 

namespace {
tflite::MicroInterpreter* global_interpreter = nullptr;
constexpr int kTensorArenaSize = 80000;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
const tflite::Model* model = ::tflite::GetModel(g_DPCRN_m34_quant_full_int_model_data);
}

int nn_setup() {
  tflite::InitializeTarget();

  model = ::tflite::GetModel(g_DPCRN_m34_quant_full_int_model_data);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION); 

  static tflite::MicroMutableOpResolver<17> micro_op_resolver;
  micro_op_resolver.AddStridedSlice();
  micro_op_resolver.AddMul();
  micro_op_resolver.AddAdd();
  micro_op_resolver.AddPad();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddUnpack();
  micro_op_resolver.AddLogistic();
  micro_op_resolver.AddSub();
  micro_op_resolver.AddTanh();
  micro_op_resolver.AddPack();
  micro_op_resolver.AddMean();
  micro_op_resolver.AddSquaredDifference();
  micro_op_resolver.AddRsqrt();
  micro_op_resolver.AddConcatenation();
  micro_op_resolver.AddTransposeConv();

  static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  global_interpreter = &static_interpreter;

  if (global_interpreter == nullptr) return 1;

  TfLiteStatus allocate_status = global_interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) return 2;

  return 0;
}

int nn_inference(int8_t* numElementsInputTensors, int32_t** inputTensors, const int32_t* inputTensorValues,
                  int8_t* numElementsOutputTensors, int32_t** outputTensors, const int32_t* outputTensorValues) {                  

  for (int i = 0; i < *numElementsInputTensors; ++i) {
    for (int j = 0; j < inputTensorValues[i]; ++j) {
      global_interpreter->input(i)->data.int8[j] = inputTensors[i][j];
    }
  }

if (kTfLiteOk != global_interpreter->Invoke()) return 1;

for (int i = 0; i < *numElementsOutputTensors; i++ ) {
  for (int j = 0; j < outputTensorValues[i]; j++ ) {
    (outputTensors[i][j]) = int8_t(global_interpreter->output(i)->data.int8[j]);
    }
  }

return 0;
} 

int wrapper_nn_setup(){
  int ret = nn_setup();
  return ret;
}

int wrapper_nn_inference(int8_t* numOfInputs, int32_t** inputTensors, const int32_t* inputTensorShapes,
                         int8_t* numOfOutputs, int32_t** outputTensors, const int32_t* outputTensorShapes) {

  int ret = nn_inference(numOfInputs, inputTensors, inputTensorShapes,
               numOfOutputs, outputTensors, outputTensorShapes);
  return ret;
}

int avatronics_test() {
  int ret = 0;
  constexpr int32_t inputTensorShapes[] = {(1*1*8*32), (1*1*257*3)};

  int32_t inputTensor_0[inputTensorShapes[0]];
  int32_t inputTensor_1[inputTensorShapes[1]];

  int32_t* inputTensors[] = {inputTensor_0, inputTensor_1};

  int8_t numOfInputs = sizeof(inputTensors) / sizeof(inputTensors[0]);

  /* Initializing whole input array equal zero */
  for (int i = 0; i < (numOfInputs); ++i) {
    for (int j = 0; j < inputTensorShapes[i]; ++j) {
        inputTensors[i][j] = 0; 
    }
  }

  /* Declaring outputs */
  constexpr int32_t outputTensorShapes[] = {(1*1*257) , (1*1*257),
                                            (1*1*8*32), (1*1*257)};

  int32_t outputTensor_0[outputTensorShapes[0]];
  int32_t outputTensor_1[outputTensorShapes[1]];
  int32_t outputTensor_2[outputTensorShapes[2]];
  int32_t outputTensor_3[outputTensorShapes[3]];
  
  int32_t *outputTensors[] = {outputTensor_0, outputTensor_1,
                              outputTensor_2, outputTensor_3};

  int8_t numOfOutputs = sizeof(outputTensors) / sizeof(outputTensors[0]);

  for (int i = 0; i < numOfOutputs; i++) {
    for (int j = 0; j < outputTensorShapes[i]; j++) {
      outputTensors[i][j] = 0;
    }
  }

  /*Calling functions to perform NN*/

  if (wrapper_nn_setup() != 0) MicroPrintf("\n\n ERROR %d !!!", ret);

  wrapper_nn_inference(&numOfInputs, inputTensors, inputTensorShapes,
                       &numOfOutputs, outputTensors, outputTensorShapes);

  /* Print the input tp NN*/
  MicroPrintf("\nInput values: (%d)", numOfInputs);
  for (int i = 0; i < numOfInputs; i++){
  MicroPrintf("\n\tInputTensor_%d: (%d) \n\t\t[",i, inputTensorShapes[i]);
    for (int j = 0; j < inputTensorShapes[i]; j++) {
      if (j % 28 == 0 && j!=0){
        MicroPrintf("\n\t\t"); 
      }
      MicroPrintf("%d,", inputTensors[i][j]);
    }
  MicroPrintf("]\n\n");
  }

  /* Prints the predections*/
  MicroPrintf("\nOutput values: (%d)",numOfOutputs);
  for (int i = 0; i < numOfOutputs; i++) {
    MicroPrintf("\n\tOutputTensor_%d: (%d) \n\t\t[", i, outputTensorShapes[i]); 
    for (int j = 0; j < outputTensorShapes[i]; j++) {
      if (j % 14 == 0 && j != 0) { 
        MicroPrintf("\n\t\t "); 
      }
      MicroPrintf("%d, ", outputTensors[i][j]);
    }
  MicroPrintf("]\n\n");
  } 
  
  MicroPrintf("\n\n~~~ALL TESTS PASSED~~~\n");
  return 0;
}