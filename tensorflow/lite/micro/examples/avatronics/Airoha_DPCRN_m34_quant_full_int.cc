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
// #include "tensorflow/lite/micro/examples/avatronics/models/DPCRN_m34_quant_full_int_model_data.h"
// #include "tensorflow/lite/micro/examples/avatronics/models/DPCRN_m34_model_data.h"

/*============ Included from GitHub folder ======================*/
#include "../../shared_ubuntu/AVAHEEAR/AVA_DPCRN/pretrained_weights/DPCRN_S/airoha_models/DPCRN_m34_model_data.h"
#include "../../shared_ubuntu/AVAHEEAR/AVA_DPCRN/pretrained_weights/DPCRN_S/airoha_models/DPCRN_m34_quant_full_int_model_data.h"

/*Enables time measurement - BUILD_TYPE should be 'release' for this to work */ 
// #define CYCLES_TAKEN

// #define MODEL_DATATYPE_DOUBLE /*Write in uppercases*/
#define MODEL_DATATYPE_INT32 /*Write in uppercases*/

#ifdef CYCLES_TAKEN
#define PROFILE
#define PROF_ALLOCATE
#include "third_party/avatronics/xt_profiler.h"
#endif

namespace {
  tflite::MicroInterpreter* global_interpreter = nullptr;
  constexpr int kTensorArenaSize = 100000;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
  const tflite::Model* model = nullptr;
}

int nn_setup() {
  tflite::InitializeTarget();

  /*  INT8    = g_DPCRN_m34_quant_full_int_model_data   */
  /*  Float32 = g_DPCRN_m34_model_data                  */
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

template <typename T>         
int nn_inference(int8_t* numElementsInputTensors, T** inputTensors, const int32_t* inputTensorValues,
                  int8_t* numElementsOutputTensors, T** outputTensors, const int32_t* outputTensorValues) {                  

#ifdef CYCLES_TAKEN
  char profiler_name[MAX_PROFILER_NAME_LENGTH];
  char profiler_params[MAX_PROFILER_PARAMS_LENGTH];
  int num_ops=0;
#endif


  for (int i = 0; i < *numElementsInputTensors; ++i) {
    for (int j = 0; j < inputTensorValues[i]; ++j) {
#if defined(MODEL_DATATYPE_INT32)
        global_interpreter->input(i)->data.int8[j] = inputTensors[i][j];
#elif defined(MODEL_DATATYPE_DOUBLE)
        global_interpreter->input(i)->data.f[j] = inputTensors[i][j];
#elif
        MicroPrintf("Unrecognized data type !");
        return 0;
#endif
    }
  }

#ifdef CYCLES_TAKEN
  XTPWR_PROFILER_OPEN(0, profiler_name, profiler_params, num_ops, "OPs/cyc", 1);
  XTPWR_PROFILER_UPDATE(0);
  XTPWR_PROFILER_START(0);
#endif

if (kTfLiteOk != global_interpreter->Invoke()) return 1;

#ifdef CYCLES_TAKEN
  XTPWR_PROFILER_STOP(0);
  XTPWR_PROFILER_UPDATE(0);
  XTPWR_PROFILER_PRINT(0);
#endif

  for (int i = 0; i < *numElementsOutputTensors; i++ ) {
    for (int j = 0; j < outputTensorValues[i]; j++ ) {
#if defined(MODEL_DATATYPE_INT32)
      outputTensors[i][j] = int8_t(global_interpreter->output(i)->data.int8[j]);
#elif defined(MODEL_DATATYPE_DOUBLE)
      outputTensors[i][j] = double(global_interpreter->output(i)->data.f[j]);
#elif
      MicroPrintf("Unrecognized data type !");
      return 0;
#endif
      }
    }

  return 0;
} 

int wrapper_nn_setup(){
  int ret = nn_setup();
  return ret;
}

template <typename T>
int wrapper_nn_inference(int8_t* numOfInputs, T** inputTensors, const int32_t* inputTensorShapes, \
                         int8_t* numOfOutputs, T** outputTensors, const int32_t* outputTensorShapes) {

    int ret = nn_inference(numOfInputs, inputTensors, inputTensorShapes, \
                           numOfOutputs, outputTensors, outputTensorShapes);
    return ret;
}

int wrapper_nn_inference_int32(int8_t* numElementsInputTensors, int32_t** inputTensors, \
                               const int32_t* inputTensorValues, int8_t* numElementsOutputTensors, \
                               int32_t** outputTensors, const int32_t* outputTensorValues) {
                                
    wrapper_nn_inference<int32_t>(numElementsInputTensors, inputTensors, inputTensorValues,
                                   numElementsOutputTensors, outputTensors, outputTensorValues);
    return 0;
}

int wrapper_nn_inference_double(int8_t* numElementsInputTensors, double** inputTensors, \
                                const int32_t* inputTensorValues, int8_t* numElementsOutputTensors, \
                                double** outputTensors, const int32_t* outputTensorValues) {

    wrapper_nn_inference<double>(numElementsInputTensors, inputTensors, inputTensorValues, \
                                 numElementsOutputTensors, outputTensors, outputTensorValues);
    return 0;
}

int avatronics_test() {
  int ret = 0;
  constexpr int32_t inputTensorShapes[] = {(1*1*8*32), (1*1*257*3)};

#if defined(MODEL_DATATYPE_INT32)
  MicroPrintf("\n\nMODEL_DATATYPE == INT32\n\n");
#elif defined(MODEL_DATATYPE_DOUBLE)
  MicroPrintf("\n\n MODEL_DATATYPE == DOUBLE\n\n");
#else
  MicroPrintf("\n\n MODEL_DATATYPE UNKNOWN\n\n");
#endif


#if defined(MODEL_DATATYPE_INT32)
  int32_t inputTensor_0[inputTensorShapes[0]];
  int32_t inputTensor_1[inputTensorShapes[1]];
#elif defined(MODEL_DATATYPE_DOUBLE)
  double inputTensor_0[inputTensorShapes[0]];
  double inputTensor_1[inputTensorShapes[1]];
#endif

#if defined(MODEL_DATATYPE_INT32)
  int32_t* inputTensors[] = {inputTensor_0, inputTensor_1};
#elif defined(MODEL_DATATYPE_DOUBLE)
  double* inputTensors[] = {inputTensor_0, inputTensor_1};
#endif

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

#if defined(MODEL_DATATYPE_INT32)
  int32_t outputTensor_0[outputTensorShapes[0]];
  int32_t outputTensor_1[outputTensorShapes[1]];
  int32_t outputTensor_2[outputTensorShapes[2]];
  int32_t outputTensor_3[outputTensorShapes[3]];
#elif defined(MODEL_DATATYPE_DOUBLE)
  double outputTensor_0[outputTensorShapes[0]];
  double outputTensor_1[outputTensorShapes[1]];
  double outputTensor_2[outputTensorShapes[2]];
  double outputTensor_3[outputTensorShapes[3]];
#endif
  
#if defined(MODEL_DATATYPE_INT32)
  int32_t *outputTensors[] = {outputTensor_0, outputTensor_1,
                              outputTensor_2, outputTensor_3};
#elif defined(MODEL_DATATYPE_DOUBLE)
  double *outputTensors[] = {outputTensor_0, outputTensor_1,
                              outputTensor_2, outputTensor_3};
#endif

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
#if defined(MODEL_DATATYPE_INT32)
      MicroPrintf("%d,", inputTensors[i][j]);
#elif defined(MODEL_DATATYPE_DOUBLE)
      MicroPrintf("%f,", inputTensors[i][j]);
#endif
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
#if defined(MODEL_DATATYPE_INT32)
      MicroPrintf("%d, ", outputTensors[i][j]);
#elif defined(MODEL_DATATYPE_DOUBLE)
      MicroPrintf("%f, ", outputTensors[i][j]);
#endif
    }
  MicroPrintf("]\n\n");
  } 
  
  MicroPrintf("\n\n~~~ALL TESTS PASSED~~~\n");
  return 0;
}