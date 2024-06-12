#ifndef CONFIG_H
#define CONFIG_H

/* ================================
 * General Configuration
 * ================================ */

/* Enables time measurement - BUILD_TYPE should be 'release' for this to work.
 *  0 to disable.
 *  1 to enable for inner loop.
 *  2 to enable for one inference. */
#define MEASURE_CYCLES_TAKEN 0

/* Enables extended log for a specfic operator. */ 
#define PRINT_XTENSA_CONV2D 1
#define PRINT_TFLM_CONV2D 1

/* Print the values of the intermediate tensors in the model.
 * Set to 1 to enable, 0 to disable. */
#define PRINT_INTERMEDIATE_TENSORS 0

#if PRINT_INTERMEDIATE_TENSORS
    /* Set to 1 to print the full tensor when PRINT_INTERMEDIATE_TENSORS is enabled.
     * Set to 0 to print a summary instead. */
    #define PRINT_FULL_TENSOR 0
#endif // PRINT_INTERMEDIATE_TENSORS

/* ================================
 * Model Datatype Configuration
 * ================================ */

/* Enables must match model datatype.
 * Set to 1 to enable, 0 to disable.
 * Note: ONLY ONE should be enabled at a time. */
#define MODEL_DATATYPE_DOUBLE 0
#define MODEL_DATATYPE_INT32 0
#define FULL_INT_INOUTS_FLOAT32 1

/* Check that only one model datatype is enabled. */
#if (MODEL_DATATYPE_DOUBLE + MODEL_DATATYPE_INT32 + FULL_INT_INOUTS_FLOAT32) > 1
    #error "Only one model datatype should be enabled at a time!"
#endif

#endif /* CONFIG_H */
