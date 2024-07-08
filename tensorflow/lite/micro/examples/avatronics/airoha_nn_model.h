#include <stdio.h>
#include <stdint.h>

extern int avatronics_test(); 

extern "C" int inputs_size();

extern "C" int outputs_size();

extern "C" int8_t nn_setup();
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

extern "C" int8_t tflm_inference(int8_t* numElementsInputTensors, double** inputTensors, const int32_t* intputTensorShapes, int8_t* numElementsOutputTensors, double** outputTensors, const int32_t* outputTensorShapes);

extern "C" int tflm_return_value_test(int8_t* numElementsInputTensors, double** inputTensors, int8_t* numElementsOutputTensors, double** outputTensors);