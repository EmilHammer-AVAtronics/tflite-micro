#include <stdio.h>
#include <stdint.h>

extern int cpp_meta_test_fct();

extern "C" int wrapper_InitializeFloatModel();

extern "C" int wrapper_LoadInput();

extern "C" int wrapper_LoadOneInput(double qCorr_24sum_from_dsp, int i);

extern "C" int wrapper_PerformInference(uint8_t* y_pred0_out1, uint8_t* y_pred1_out1);

extern "C" int wrapper_freeMemory();

extern "C" int wrapper_PerformAll(uint8_t* y_pred0_out1, uint8_t* y_pred1_out1);
 
 extern "C" int wrapper_nn_setup(); 

 extern "C" int wrapper_nn_inference(uint8_t* y_pred0_out1, uint8_t* y_pred1_out1);
 