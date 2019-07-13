
#ifndef TRTINFERENCE_H
#define TRTINFERENCE_H

namespace trt_ros_inference
{

#define USE_32BIT_FLOAT 32
#define USE_BIT_FLOAT 16

#define FLOAT_TYPE USE_32BIT_FLOAT


#define NUM_OF_INPUTS  1
#define NUM_OF_OUTPUTS 2
#define TOTAL_BINDINGS (NUM_OF_INPUTS + NUM_OF_OUTPUTS)  // Total number of inputs+outputs

#define INPUT_BINDING_NAME "input_1"
#define OUTPUT_1_BINDING_NAME "activation_8/Sigmoid"
#define OUTPUT_2_BINDING_NAME "dense_1/BiasAdd"

static int INPUT_H;
static int INPUT_W;
static const int INPUT_C = 1; // We expect grayscale image
static const int OUTPUT_ELT_SZ[NUM_OF_OUTPUTS] = {1, 1}; // Number of elements for each output
#undef MORE_THAN_1_ELEM_PER_OUTPUT  // if you change the {1, 1} of OUTPUT_ELT_SZ, change this as well. check the code for implications.

int build_engine(std::string uff_path, uint8_t input_shape[2], void** out_engine, void** out_context);
void delete_engine(void *p_engine_arg, void *p_context);
int inference(void* p_engine, void* p_context, float *input_img, float output_arr[NUM_OF_OUTPUTS]);

} // namespace trt_ros_inference

#endif // TRTINFERENCE_H
