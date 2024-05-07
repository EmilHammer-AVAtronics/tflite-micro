#include <stdio.h>
#include <stdint.h>

extern int cpp_meta_test_fct();
 
 extern "C" int wrapper_meta_setup(); 

 extern "C" int wrapper_meta_inference(uint8_t* y_pred0_out1, uint8_t* y_pred1_out1);
 