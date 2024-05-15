#ifndef CONFIG_H
#define CONFIG_H

/* ================================
 * General Configuration
 * ================================ */

/*Enables time measurement - BUILD_TYPE should be 'release' for this to work 
 *  0 to disable.
 *  1 to enable for micro_interpreter_graph.cc
 *  2 to enable for Airoha_DPCRN_m34_quant_full_int.cc */ 
#define MEASURE_CYCLES_TAKEN 2

/*Print the values of the intermediate tensors in the model
 * Set to 1 to enable, 0 to disable. */
#define PRINT_INTERMEDIATE_TENSORS 1

/*Enables must match model datatype
 * Set to 1 to enable, 0 to disable.
 * OBS. ONLY ONE AT A TIME*/ 
#define MODEL_DATATYPE_DOUBLE 0
#define MODEL_DATATYPE_INT32 1

#endif /* CONFIG_H */