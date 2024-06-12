/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/micro_interpreter_graph.h"

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/micro/examples/avatronics/config.h"

#include "tensorflow/lite/micro/kernels/kernel_util.h"

#if MEASURE_CYCLES_TAKEN == 1
#define MEASURE_CYCLES 1
#else 
#define MEASURE_CYCLES 0
#endif // MEASURE_CYCLES_TAKEN


#if MEASURE_CYCLES
#define PROFILE
#define PROF_ALLOCATE
#include "third_party/avatronics/xt_profiler.h"
#endif // MEASURE_CYCLES


#if MEASURE_CYCLES
char profiler_name[MAX_PROFILER_NAME_LENGTH];
char profiler_params[MAX_PROFILER_PARAMS_LENGTH];
int num_ops=0;
#endif // MEASURE_CYCLES


namespace tflite {
namespace {

const char* TfLiteTypeName(TfLiteType type) {
    switch (type) {
        case kTfLiteNoType: return "kTfLiteNoType";
        case kTfLiteFloat32: return "kTfLiteFloat32";
        case kTfLiteInt32: return "kTfLiteInt32";
        case kTfLiteUInt8: return "kTfLiteUInt8";
        case kTfLiteInt64: return "kTfLiteInt64";
        case kTfLiteString: return "kTfLiteString";
        case kTfLiteBool: return "kTfLiteBool";
        case kTfLiteInt16: return "kTfLiteInt16";
        case kTfLiteComplex64: return "kTfLiteComplex64";
        case kTfLiteInt8: return "kTfLiteInt8";
        case kTfLiteFloat16: return "kTfLiteFloat16";
        case kTfLiteFloat64: return "kTfLiteFloat64";
        case kTfLiteComplex128: return "kTfLiteComplex128";
        case kTfLiteUInt64: return "kTfLiteUInt64";
        case kTfLiteResource: return "kTfLiteResource";
        case kTfLiteVariant: return "kTfLiteVariant";
        case kTfLiteUInt32: return "kTfLiteUInt32";
        case kTfLiteUInt16: return "kTfLiteUInt16";
        case kTfLiteInt4: return "kTfLiteInt4";
        case kTfLiteBFloat16: return "kTfLiteBFloat16";
        default: return "UnknownType";
    }
}

const char* OpNameFromRegistration(const TFLMRegistration* registration) {
  if (registration->builtin_code == BuiltinOperator_CUSTOM) {
    return registration->custom_name;
  } else {
    return EnumNameBuiltinOperator(BuiltinOperator(registration->builtin_code));
  }
}

}  // namespace

MicroInterpreterGraph::MicroInterpreterGraph(
    TfLiteContext* context, const Model* model, MicroAllocator* allocator,
    MicroResourceVariables* resource_variables)
    : context_(context),
      model_(model),
      allocator_(allocator),
      current_subgraph_index_(0),
      resource_variables_(resource_variables) {
  if (model != nullptr) {
    subgraphs_ = model->subgraphs();
  }
}

MicroInterpreterGraph::~MicroInterpreterGraph() {}

TfLiteStatus MicroInterpreterGraph::InitSubgraphs() {
  int previous_subgraph_idx = current_subgraph_index_;

  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
    for (size_t i = 0; i < operators_size; ++i) {
      TfLiteNode* node =
          &(subgraph_allocations_[subgraph_idx].node_and_registrations[i].node);
      const TFLMRegistration* registration = subgraph_allocations_[subgraph_idx]
                                                 .node_and_registrations[i]
                                                 .registration;
      size_t init_data_size;
      const char* init_data;
      if (registration->builtin_code == BuiltinOperator_CUSTOM) {
        init_data = reinterpret_cast<const char*>(node->custom_initial_data);
        init_data_size = node->custom_initial_data_size;
      } else {
        init_data = reinterpret_cast<const char*>(node->builtin_data);
        init_data_size = 0;
      }
      if (registration->init) {
        node->user_data =
            registration->init(context_, init_data, init_data_size);
      }
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;

  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterGraph::PrepareSubgraphs() {
  int previous_subgraph_idx = current_subgraph_index_;

  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
    for (size_t i = 0; i < operators_size; ++i) {
      TfLiteNode* node =
          &(subgraph_allocations_[subgraph_idx].node_and_registrations[i].node);
      const TFLMRegistration* registration = subgraph_allocations_[subgraph_idx]
                                                 .node_and_registrations[i]
                                                 .registration;
      if (registration->prepare != nullptr) {
        TfLiteStatus prepare_status = registration->prepare(context_, node);
        if (prepare_status != kTfLiteOk) {
          MicroPrintf("Node %s (number %df) failed to prepare with status %d",
                      OpNameFromRegistration(registration), i, prepare_status);
          return kTfLiteError;
        }
      }
      allocator_->FinishPrepareNodeAllocations(/*node_id=*/i);
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;

  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterGraph::ResetSubgraphs() {
  int previous_subgraph_idx = current_subgraph_index_;

  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
    for (size_t i = 0; i < operators_size; ++i) {
      TfLiteNode* node =
          &(subgraph_allocations_[subgraph_idx].node_and_registrations[i].node);
      const TFLMRegistration* registration = subgraph_allocations_[subgraph_idx]
                                                 .node_and_registrations[i]
                                                 .registration;
      // registration is allocated outside the interpreter, so double check to
      // make sure it's not nullptr;
      if (registration != nullptr && registration->reset != nullptr) {
        registration->reset(context_, node->user_data);
      }
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;

  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterGraph::FreeSubgraphs() {
  int previous_subgraph_idx = current_subgraph_index_;

  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
    for (size_t i = 0; i < operators_size; ++i) {
      TfLiteNode* node =
          &(subgraph_allocations_[subgraph_idx].node_and_registrations[i].node);
      const TFLMRegistration* registration = subgraph_allocations_[subgraph_idx]
                                                 .node_and_registrations[i]
                                                 .registration;
      // registration is allocated outside the interpreter, so double check to
      // make sure it's not nullptr;
      if (registration != nullptr && registration->free != nullptr) {
        registration->free(context_, node->user_data);
      }
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;

  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterGraph::InvokeSubgraph(int subgraph_idx) {
  int previous_subgraph_idx = current_subgraph_index_;
  current_subgraph_index_ = subgraph_idx;

  if (static_cast<size_t>(subgraph_idx) >= subgraphs_->size()) {
    MicroPrintf("Accessing subgraph %d but only %d subgraphs found",
                subgraph_idx, subgraphs_->size());
    return kTfLiteError;
  }
  uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
  for (size_t i = 0; i < operators_size; ++i) {
#if PRINT_INTERMEDIATE_TENSORS
#endif // PRINT_INTERMEDIATE_TENSORS
#if MEASURE_CYCLES
    XTPWR_PROFILER_OPEN(0, profiler_name, profiler_params, num_ops, "OPs/cyc", 1);
    XTPWR_PROFILER_UPDATE(0);
    XTPWR_PROFILER_START(0);
#endif //MEASURE_CYCLES

    TfLiteNode* node =
        &(subgraph_allocations_[subgraph_idx].node_and_registrations[i].node);
    const TFLMRegistration* registration = subgraph_allocations_[subgraph_idx]
                                               .node_and_registrations[i]
                                               .registration;

// This ifdef is needed (even though ScopedMicroProfiler itself is a no-op with
// -DTF_LITE_STRIP_ERROR_STRINGS) because the function OpNameFromRegistration is
// only defined for builds with the error strings.
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
    ScopedMicroProfiler scoped_profiler(
        OpNameFromRegistration(registration),
        reinterpret_cast<MicroProfilerInterface*>(context_->profiler));
#endif

    TFLITE_DCHECK(registration->invoke);
    TfLiteStatus invoke_status = registration->invoke(context_, node);

    // All TfLiteTensor structs used in the kernel are allocated from temp
    // memory in the allocator. This creates a chain of allocations in the
    // temp section. The call below resets the chain of allocations to
    // prepare for the next call.
    allocator_->ResetTempAllocations();

    if (invoke_status == kTfLiteError) {
      MicroPrintf("Node %s (number %d) failed to invoke with status %d",
                  OpNameFromRegistration(registration), i, invoke_status);
      return kTfLiteError;
    } else if (invoke_status != kTfLiteOk) {
      return invoke_status;
    }

#if MEASURE_CYCLES
    XTPWR_PROFILER_STOP(0);
    XTPWR_PROFILER_UPDATE(0);
    XTPWR_PROFILER_PRINT(0);
#endif // MEASURE_CYCLES

#if PRINT_INTERMEDIATE_TENSORS

    int32_t output_size = node->outputs->size;
    int32_t input_size = node->inputs->size;
#if MODEL_DATATYPE_INT32 || FULL_INT_INOUTS_FLOAT32
    int* output_tensors = node->outputs->data;
    int* input_tensors = node->inputs->data;
#elif MODEL_DATATYPE_DOUBLE
    double* output_tensors = node->outputs->data;
    double* input_tensors = node->inputs->data;
#endif
    MicroPrintf("OP Index: %d (%s) \n",i, OpNameFromRegistration(registration));
    MicroContext* micro_context = GetMicroContext(context_);  
    MicroPrintf("\tNumber of inputs %d:\n", input_size);

    for (int j = 0; j < input_size; j++) {
        MicroPrintf("\tTensor %d -->\tInput(%d)\n", *(input_tensors + j), j);
    }

    for (int j = 0; j < input_size; j++) {
        MicroPrintf("\n\t\tTensor %d:\n", *(input_tensors + j));
        if (*(input_tensors + j) == -1) {
            MicroPrintf("\t\t\t\tSKIPS WHEN OP -1 !!\n");
        } else {
          TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, j);
          TF_LITE_ENSURE(context_, input != nullptr);
          RuntimeShape input_shape = GetTensorShape(input);
          int32_t num_dims = input_shape.DimensionsCount();
          TfLiteType type = input->type;
          MicroPrintf("\t\t\tDatatype: %s \n", TfLiteTypeName(type));
          MicroPrintf("\t\t\tNumber of dimensions: %d\n", num_dims);
          MicroPrintf("\t\t\tShape: [");
          if (0 == num_dims) MicroPrintf("]\n"); 

          for (int k = 0; k < num_dims; k++) {
              if (k == (num_dims - 1))  {
                  MicroPrintf("%d]\n", input_shape.Dims(k));
              } else {
                  MicroPrintf("%d, ", input_shape.Dims(k));
              }
          }
          MicroPrintf("\n\t\t\tinput_tensors: %d", *(input_tensors + j));
          for (int k = 0; k < num_dims; k++) {
              int stride = 1;

              /* calculates the stride for dimension k*/
              for (int p= k+1; p < num_dims; p++){
                stride *= input_shape.Dims(p);
              }

              MicroPrintf("\n\t\t\tValues of dim %d, of size %d:", k, input_shape.Dims(k));
              MicroPrintf("\n\t\t\tStride of %d\n\t\t\t\t[", stride);

              for (int l = 0; l < input_shape.Dims(k); l++) {
                  if (l % 14 == 0 && l != 0) {
                      MicroPrintf("\n\t\t\t\t ");
                  }
                  if (l == (input_shape.Dims(k) - 1)) {
                      if (type == kTfLiteInt8) MicroPrintf("%d]\n", input->data.int8[l*stride]);
                      if (type == kTfLiteInt32) MicroPrintf("%d]\n", input->data.i32[l*stride]);
                      if (type == kTfLiteFloat32) MicroPrintf("%f]\n", double(input->data.f[l*stride]));
                  } else {
                      if (type == kTfLiteInt8) MicroPrintf("%d, ", input->data.int8[l*stride]);
                      if (type == kTfLiteInt32) MicroPrintf("%d, ", input->data.i32[l*stride]);
                      if (type == kTfLiteFloat32) MicroPrintf("%f, ", double(input->data.f[l*stride]));
                  }
              }
          }
#if PRINT_FULL_TENSOR
            /*Printing all tensor values */
            int total_numbers_of_values = 1;
            for (int k = 0; k < num_dims; k++) {
                  total_numbers_of_values *= input_shape.Dims(k);
              }
            
            int count_greater_than_one = 0;
            for (int k = 0; k < num_dims; k++) {
                if (input_shape.Dims(k) > 1) {
                    count_greater_than_one++;
                }
            }

            if (count_greater_than_one >= 2) {
            MicroPrintf("\n\n\t\t\tPrinting all tensor values:");
            MicroPrintf("\n\t\t\tTotal numbers of values: %d\n\t\t\t\t[", total_numbers_of_values);

            for (int l = 0; l < total_numbers_of_values; l++) {
                if (l % 14 == 0 && l != 0) {
                    MicroPrintf("\n\t\t\t\t "); 
                }
                if (l == (total_numbers_of_values - 1)) {
                    if (type == kTfLiteInt8) MicroPrintf("%d]\n", input->data.int8[l]);
                    if (type == kTfLiteInt32) MicroPrintf("%d]\n", input->data.i32[l]);
                    if (type == kTfLiteFloat32)MicroPrintf("%f]\n", double(input->data.f[l]));
                } else {
                    if (type == kTfLiteInt8) MicroPrintf("%d, ", input->data.int8[l]);
                    if (type == kTfLiteInt32) MicroPrintf("%d, ", input->data.i32[l]);
                    if (type == kTfLiteFloat32)MicroPrintf("%f, ", double(input->data.f[l]));
                  }
              }
            } 
#endif // PRINT_FULL_TENSOR
          micro_context->DeallocateTempTfLiteTensor(input);
        }
    }

  /* OUTPUT */
    MicroPrintf("\tNumber of outputs %d:\n", output_size);
    for (int j = 0; j < output_size; j++) {
        MicroPrintf("\tOutput(%d) -->\tTensor %d\n", j, *(output_tensors + j));
    }

    for (int j = 0; j < output_size; j++) {
        MicroPrintf("\n\t\tTensor %d:\n", *(output_tensors + j));
        if (*(output_tensors + j) == -1) {
            MicroPrintf("\t\t\t\tSKIPS WHEN OP -1 !!\n");
        } else {
            TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, j);
            TF_LITE_ENSURE(context_, output != nullptr);
            RuntimeShape output_shape = GetTensorShape(output);
            int32_t num_dims = output_shape.DimensionsCount();
            TfLiteType type = output->type;
            MicroPrintf("\t\t\tDatatype: %s \n", TfLiteTypeName(type));
            MicroPrintf("\t\t\tNumber of dimensions: %d\n", num_dims);
            MicroPrintf("\t\t\tShape:\t[");
            if (0 == num_dims) MicroPrintf("]\n"); 

            for (int k = 0; k < num_dims; k++) {
                if (k == (num_dims - 1)) {
                    MicroPrintf("%d]\n", output_shape.Dims(k));
                } else {
                    MicroPrintf("%d, ", output_shape.Dims(k));
                }
            }
            MicroPrintf("\n\t\t\tOutput_tensors: %d", *(output_tensors + j));
            for (int k = 0; k < num_dims; k++) {
              int stride = 1;

              /* calculates the stride for dimension k*/
              for (int p= k+1; p < num_dims; p++){
                stride *= output_shape.Dims(p);
              }

              // Calculate the index in the flat array
              MicroPrintf("\n\t\t\tValues of dim %d, of size %d:", k, output_shape.Dims(k));
              MicroPrintf("\n\t\t\tStride: %d\n\t\t\t\t[", stride);

                for (int l = 0; l < output_shape.Dims(k); l++) {
                    if (l % 14 == 0 && l != 0) {
                        MicroPrintf("\n\t\t\t\t ");
                    }
                    if (l == (output_shape.Dims(k) - 1)) {
                        if (type == kTfLiteInt8) MicroPrintf("%d]\n", output->data.int8[l*stride]);
                        if (type == kTfLiteFloat32) MicroPrintf("%f]\n", double(output->data.f[l*stride]));
                    } else {
                        if (type == kTfLiteInt8) MicroPrintf("%d, ", output->data.int8[l*stride]);
                        if (type == kTfLiteFloat32) MicroPrintf("%f, ", double(output->data.f[l*stride]));
                    }
                }
            }
#if PRINT_FULL_TENSOR
              /*Printing all tensor values */
              int total_numbers_of_values = 1;
              for (int k = 0; k < num_dims; k++) {
                    total_numbers_of_values *= output_shape.Dims(k);
                }
              
              int count_greater_than_one = 0;
              for (int k = 0; k < num_dims; k++) {
                  if (output_shape.Dims(k) > 1) {
                      count_greater_than_one++;
                  }
              }

              if (count_greater_than_one >= 2) {
              MicroPrintf("\n\n\t\t\tPrinting all tensor values:");


              MicroPrintf("\n\t\t\tTotal numbers of values: %d\n\t\t\t\t[", total_numbers_of_values);

              for (int l = 0; l < total_numbers_of_values; l++) {
                  if (l % 14 == 0 && l != 0) {
                      MicroPrintf("\n\t\t\t\t ");
                  }
                  if (l == (total_numbers_of_values - 1)) {
                    if (type == kTfLiteInt8)  MicroPrintf("%d]\n", output->data.int8[l]);
                    if (type == kTfLiteFloat32) MicroPrintf("%f]\n", double(output->data.f[l]));
                  } else {
                    if (type == kTfLiteInt8) MicroPrintf("%d, ", output->data.int8[l]);
                    if (type == kTfLiteFloat32) MicroPrintf("%f, ", double(output->data.f[l]));
                   }
                }
              }
#endif // PRINT_FULL_TENSOR
            micro_context->DeallocateTempTfLiteTensor(output);
        }
    }
#endif // PRINT_INTERMEDIATE_TENSORS 

  }
  current_subgraph_index_ = previous_subgraph_idx;


  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterGraph::ResetVariableTensors() {
  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    const SubGraph* subgraph = (*subgraphs_)[subgraph_idx];
    for (size_t i = 0; i < subgraph->tensors()->size(); ++i) {
      auto* tensor = subgraph->tensors()->Get(i);
      if (tensor->is_variable()) {
        size_t buffer_size;
        TF_LITE_ENSURE_STATUS(TfLiteEvalTensorByteLength(
            &subgraph_allocations_[subgraph_idx].tensors[i], &buffer_size));

        int value = 0;
        if (tensor->type() == tflite::TensorType_INT8) {
          value = tensor->quantization()->zero_point()->Get(0);
        }
        memset(subgraph_allocations_[subgraph_idx].tensors[i].data.raw, value,
               buffer_size);
      }
    }
  }
  if (resource_variables_ != nullptr) {
    resource_variables_->ResetAll();
  }

  return kTfLiteOk;
}

int MicroInterpreterGraph::NumSubgraphs() {
  return model_->subgraphs()->size();
}

void MicroInterpreterGraph::SetSubgraphAllocations(
    SubgraphAllocations* subgraph_allocations) {
  subgraph_allocations_ = subgraph_allocations;
}

size_t MicroInterpreterGraph::NumSubgraphInputs(int subgraph_idx) {
  return model_->subgraphs()->Get(subgraph_idx)->inputs()->size();
}

TfLiteEvalTensor* MicroInterpreterGraph::GetSubgraphInput(int subgraph_idx,
                                                          int input_idx) {
  int tensor_idx =
      model_->subgraphs()->Get(subgraph_idx)->inputs()->Get(input_idx);
  return &subgraph_allocations_[subgraph_idx].tensors[tensor_idx];
}

size_t MicroInterpreterGraph::NumSubgraphOutputs(int subgraph_idx) {
  return model_->subgraphs()->Get(subgraph_idx)->outputs() == nullptr
             ? 0
             : model_->subgraphs()->Get(subgraph_idx)->outputs()->size();
}

TfLiteEvalTensor* MicroInterpreterGraph::GetSubgraphOutput(int subgraph_idx,
                                                           int output_idx) {
  int tensor_idx =
      model_->subgraphs()->Get(subgraph_idx)->outputs()->Get(output_idx);
  return &subgraph_allocations_[subgraph_idx].tensors[tensor_idx];
}

}  // namespace tflite
