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

#include "airoha_nn_model.h"
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

/*============================== CONFIG =======================================*/
#include "tensorflow/lite/micro/examples/avatronics/config.h"


/*========== LOCAL MODELS TO AIROHA =================*/
#include "tensorflow/lite/micro/examples/avatronics/models/DPCRN_m71_quant_full_int_InOuts_float32_13_6_2024_model_data.h"
// #include "tensorflow/lite/micro/examples/avatronics/models/DPCRN_m71_quant_full_int_InOuts_float32_model_data.h"

/*======================== Included from GitHub folder ==========================*/
// #include "../../shared_ubuntu/AVAHEEAR/AVA_DPCRN/pretrained_weights/DPCRN_S/airoha_models/DPCRN_m62_quant_full_int_InOuts_float32_7_6_2024_model_data.h"
// #include "../../shared_ubuntu/AVAHEEAR/AVA_DPCRN/pretrained_weights/DPCRN_S/airoha_models/DPCRN_m71_quant_full_int_InOuts_float32_13_6_2024_model_data.h"
// #include "../../shared_ubuntu/AVAHEEAR/AVA_DPCRN/pretrained_weights/DPCRN_S/airoha_models/DPCRN_m71_quant_full_int_InOuts_float32_model_data.h"

#if MEASURE_CYCLES_TAKEN == 2
#define MEASURE_CYCLES 1
#else 
#define MEASURE_CYCLES 0 
#endif //MEASURE_CYCLES_TAKEN

#if MEASURE_CYCLES
#define PROFILE
#define PROF_ALLOCATE
#include "third_party/avatronics/xt_profiler.h"
#endif //MEASURE_CYCLES

namespace {
  tflite::MicroInterpreter* global_interpreter = nullptr;
  constexpr int kTensorArenaSize = 76000;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
  const tflite::Model* model = nullptr;
}
int nn_setup() {
  tflite::InitializeTarget();

  model = ::tflite::GetModel(g_DPCRN_m71_quant_full_int_InOuts_float32_13_6_2024_model_data);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  static tflite::MicroMutableOpResolver<19> micro_op_resolver;
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddStridedSlice();
  micro_op_resolver.AddPad();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddUnpack();
  micro_op_resolver.AddLogistic();
  micro_op_resolver.AddMul();
  micro_op_resolver.AddSub();
  micro_op_resolver.AddAdd();
  micro_op_resolver.AddTanh();
  micro_op_resolver.AddPack();
  micro_op_resolver.AddMean();
  micro_op_resolver.AddSquaredDifference();
  micro_op_resolver.AddRsqrt();
  micro_op_resolver.AddConcatenation();
  micro_op_resolver.AddDequantize();
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

#if MEASURE_CYCLES
  char profiler_name[MAX_PROFILER_NAME_LENGTH];
  char profiler_params[MAX_PROFILER_PARAMS_LENGTH];
  int num_ops=0;
#endif // MEASURE_CYCLES

  for (int i = 0; i < *numElementsInputTensors; ++i) {
    for (int j = 0; j < inputTensorValues[i]; ++j) {
#if MODEL_DATATYPE_INT32
        global_interpreter->input(i)->data.int8[j] = inputTensors[i][j];
#elif MODEL_DATATYPE_DOUBLE || FULL_INT_INOUTS_FLOAT32
        global_interpreter->input(i)->data.f[j] = inputTensors[i][j];
#elif
        MicroPrintf("Unrecognized data type !");
        return 0;
#endif
    }
  }

#if MEASURE_CYCLES
  XTPWR_PROFILER_OPEN(0, profiler_name, profiler_params, num_ops, "OPs/cyc", 1);
  XTPWR_PROFILER_UPDATE(0);
  XTPWR_PROFILER_START(0);
#endif // MEASURE_CYCLES

if (kTfLiteOk != global_interpreter->Invoke()) return 1;

#if MEASURE_CYCLES
  XTPWR_PROFILER_STOP(0);
  XTPWR_PROFILER_UPDATE(0);
  XTPWR_PROFILER_PRINT(0);
#endif // MEASURE_CYCLES

  for (int i = 0; i < *numElementsOutputTensors; i++ ) {
    for (int j = 0; j < outputTensorValues[i]; j++ ) {
#if MODEL_DATATYPE_INT32
      outputTensors[i][j] = int8_t(global_interpreter->output(i)->data.int8[j]);
#elif MODEL_DATATYPE_DOUBLE || FULL_INT_INOUTS_FLOAT32
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


int inputs_size(){
  return global_interpreter->inputs_size();
}

int outputs_size(){
  return global_interpreter->outputs_size();
}

int initialization_status(){
  return global_interpreter->initialization_status();
}

int tflm_inference(int8_t* numElementsInputTensors, double** inputTensors, const int32_t* inputTensorShapes,
              int8_t* numElementsOutputTensors, double** outputTensors, const int32_t* outputTensorShapes){
if (kTfLiteOk != global_interpreter->initialization_status()) return 1;

  for (int i = 0; i < *numElementsInputTensors; ++i) {
    for (int j = 0; j < static_cast<int>(inputTensorShapes[i]); ++j) {
      if (j < static_cast<int>(global_interpreter->input(i)->bytes / sizeof(float))) { // test for out-of bounds
        global_interpreter->input(i)->data.f[j] = inputTensors[i][j];
      } else {
        return 2; // out of bound
      }
    }
  }

  TfLiteStatus status = global_interpreter->Invoke();
  if (status != kTfLiteOk) {
    return 3;
  }

  for (int i = 0; i < *numElementsOutputTensors; i++ ) {
    for (int j = 0; j < static_cast<int>(outputTensorShapes[i]); j++ ) {
     if (j < static_cast<int>(global_interpreter->output(i)->bytes / sizeof(float))) { // test for out-of bounds
        outputTensors[i][j] = static_cast<double>(global_interpreter->output(i)->data.f[j]);
      } else {
        return 4; // out of bound
      }
    }
  }
  return 0;
}

int tflm_return_value_test(int8_t* numElementsInputTensors, double** inputTensors,
                           int8_t* numElementsOutputTensors, double** outputTensors){
  /*Input*/
  *numElementsInputTensors = 1;
  inputTensors[0][0] = 2;

  /*Output*/
  *numElementsOutputTensors = 3;
  outputTensors[0][0] = 4;
  return 0;
}

int avatronics_test() {
  constexpr int32_t inputTensorShapes[] = {(1*0*8*32),
                                           (1*1*257*3)
                                           };

#if MODEL_DATATYPE_INT32
  int32_t inputTensor_0[inputTensorShapes[0]];
  int32_t inputTensor_1[inputTensorShapes[1]];
#elif MODEL_DATATYPE_DOUBLE || FULL_INT_INOUTS_FLOAT32
  double inputTensor_0[inputTensorShapes[0]];
  double inputTensor_1[inputTensorShapes[1]];
#endif

#if MODEL_DATATYPE_INT32
  int32_t* inputTensors[] = {inputTensor_0, inputTensor_1};
#elif MODEL_DATATYPE_DOUBLE || FULL_INT_INOUTS_FLOAT32
  double* inputTensors[] = {inputTensor_0, inputTensor_1};
#endif

  int8_t numElementsInputTensors = sizeof(inputTensors) / sizeof(inputTensors[0]);
  
  /* Initializing whole input array equal zero */
  int counter = 0;
  for (int i = 0; i < numElementsInputTensors; ++i) {
    for (int j = 0; j < inputTensorShapes[i]; ++j) {
        inputTensors[i][j] = counter;
    }
  }

  /* Declaring outputs */
  constexpr int32_t outputTensorShapes[] = {(1*1*257),
                                            // (1*1*8*32),
                                            (1*1*257),
                                            (1*1*257)
                                            };

#if MODEL_DATATYPE_INT32
  int32_t outputTensor_0[outputTensorShapes[0]];
  int32_t outputTensor_1[outputTensorShapes[1]];
  int32_t outputTensor_2[outputTensorShapes[2]];
  // int32_t outputTensor_3[outputTensorShapes[3]];
#elif MODEL_DATATYPE_DOUBLE || FULL_INT_INOUTS_FLOAT32
  double outputTensor_0[outputTensorShapes[0]];
  double outputTensor_1[outputTensorShapes[1]];
  double outputTensor_2[outputTensorShapes[2]];
  // double outputTensor_3[outputTensorShapes[3]];
#endif
  
#if MODEL_DATATYPE_INT32
  int32_t *outputTensors[] = {outputTensor_0,
                              outputTensor_1,
                              outputTensor_2,
                            // outputTensor_3
                              };
#elif MODEL_DATATYPE_DOUBLE || FULL_INT_INOUTS_FLOAT32
  double *outputTensors[] = {outputTensor_0,
                             outputTensor_1,
                             outputTensor_2,
                            //  outputTensor_3
                             };
#endif

int8_t numElementsOutputTensors = sizeof(outputTensors) / sizeof(outputTensors[0]);
  for (int i = 0; i < numElementsOutputTensors; i++) {
    for (int j = 0; j < outputTensorShapes[i]; j++) {
      outputTensors[i][j] = 0;
    }
  }

  /*Test wrapper_nn_setup*/
  if (wrapper_nn_setup()){
    MicroPrintf("\nWRAPPER_NN_SETUP FAILS!\n");
    return 0;
  } 

MicroPrintf("\nArena_used_bytes: %d\n", global_interpreter->arena_used_bytes());

  /*Test return_value_test*/
#if TEST_RETURN_VALUE
      tflm_return_value_test(&numElementsInputTensors, inputTensors,
                            &numElementsOutputTensors, outputTensors);
      
      if (numElementsInputTensors != 1) {
          MicroPrintf("\nnumElementsInputTensors != 1!\n");
          return 1;
      }
      if (inputTensors[0][0] != 2) {
          MicroPrintf("\ninputTensors[0][0] != 2!\n");
          return 2;
      }
      if (numElementsOutputTensors != 3) {
          MicroPrintf("\nnumElementsOutputTensors != 3!\n");
          return 3;
      }
      if (outputTensors[0][0] != 4) {
          MicroPrintf("\noutputTensors[0][0] != 4!\n");
          return 4;
      }
      MicroPrintf("\nRETURN_VALUE_TEST SUCCESS!\n");
#endif
  
  /*Test wrapper_nn_inference*/
  if (wrapper_nn_inference(&numElementsInputTensors, inputTensors, inputTensorShapes,
                           &numElementsOutputTensors, outputTensors, outputTensorShapes)) {
    MicroPrintf("\nWRAPPER_NN_INFERENCE FAILS!\n");
    return 0;
  }

  /* Print the input tp NN*/
  MicroPrintf("\nInput values: (%d)", numElementsInputTensors);
  for (int i = 0; i < numElementsInputTensors; i++){
  MicroPrintf("\n\tInputTensor_%d: (%d) \n\t\t[",i, inputTensorShapes[i]);
    for (int j = 0; j < inputTensorShapes[i]; j++) {
      if (j % 28 == 0 && j!=0){
        MicroPrintf("\n\t\t"); 
      }
#if MODEL_DATATYPE_INT32
      MicroPrintf("%d,", inputTensors[i][j]);
#elif MODEL_DATATYPE_DOUBLE || FULL_INT_INOUTS_FLOAT32
      MicroPrintf("%f,", inputTensors[i][j]);
#endif
    } 
  MicroPrintf("]\n\n");
  }

  /* Prints the predections*/
  MicroPrintf("\nOutput values: (%d)",numElementsOutputTensors);
  for (int i = 0; i < numElementsOutputTensors; i++) {
    MicroPrintf("\n\tOutputTensor_%d: (%d) \n\t\t[", i, outputTensorShapes[i]);
    for (int j = 0; j < outputTensorShapes[i]; j++) {
      if (j % 14 == 0 && j != 0) { 
        MicroPrintf("\n\t\t "); 
      }
#if MODEL_DATATYPE_INT32
      MicroPrintf("%d, ", outputTensors[i][j]);
#elif MODEL_DATATYPE_DOUBLE || FULL_INT_INOUTS_FLOAT32
      MicroPrintf("%f, ", outputTensors[i][j]);
#endif
    }
  MicroPrintf("]\n\n");
  } 
  
  MicroPrintf("\n\n~~~ALL TESTS PASSED~~~\n");
  return 0;
}