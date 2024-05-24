#include <stdio.h>
#include <stdint.h>

extern int avatronics_test(); 
 
extern "C" int wrapper_nn_setup();

/**
 * Perform neural network inference using the provided input and output tensors.
 * @param numOfInputs Pointer to a variable holding the number of input tensors.
 * @param inputTensors Pointer to an array of pointers to input tensors.
 * @param inputTensorShapes Pointer to an array holding the shapes of input tensors.
 * @param numOfOutputs Pointer to a variable holding the number of output tensors.
 * @param outputTensors Pointer to an array of pointers to output tensors.
 * @param outputTensorShapes Pointer to an array holding the shapes of output tensors.
 * @return Status code indicating the success or failure of the inference process.
 *         0 indicates success, while other values indicate errors.
 */ 

extern "C" int wrapper_nn_inference_int32(int8_t* numElementsInputTensors, int32_t** inputTensors, const int32_t* inputTensorValues, int8_t* numElementsOutputTensors, int32_t** outputTensors, const int32_t* outputTensorValues);

extern "C" int wrapper_nn_inference_double(int8_t* numElementsInputTensors, double** inputTensors, const int32_t* inputTensorValues, int8_t* numElementsOutputTensors, double** outputTensors, const int32_t* outputTensorValues);

extern "C" int wrapper_nn_inference_test(int8_t* numElementsInputTensors, double** inputTensors, const int32_t* inputTensorValues, int8_t* numElementsOutputTensors, double** outputTensors, const int32_t* outputTensorValues);

extern "C" int return_value_test(int8_t* numElementsInputTensors, double** inputTensors);