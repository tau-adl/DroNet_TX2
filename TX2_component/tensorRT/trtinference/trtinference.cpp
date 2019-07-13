#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>
#include "NvInfer.h"
#include "NvUffParser.h"

#include "NvUtils.h"

using namespace nvuffparser;
using namespace nvinfer1;
#include "common.h"

#include "trtinference.h"

#define DEBUG
#ifdef DEBUG
#define LOG(fmt_, ...) printf((fmt_), ##__VA_ARGS__)
#else
#define LOG(fmt_, ...)
#endif

#define DEBUG_IMAGE_VALUES

namespace trt_ros_inference
{

static Logger gLogger;

#define MAX_WORKSPACE (1 << 30)

#define RETURN_AND_LOG(ret, severity, message)                                              \
    do {                                                                                    \
        std::string error_message = "dronet_app: " + std::string(message);            \
        gLogger.log(ILogger::Severity::k ## severity, error_message.c_str());               \
        return (ret);                                                                       \
    } while(0)


static inline int64_t volume(const Dims& d)
{
        int64_t v = 1;
        for (int64_t i = 0; i < d.nbDims; i++)
                v *= d.d[i];
        return v;
}

/* TODO erase
static std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"tensorRT/dronet/"};
    return locateFile(input,dirs);
}
*/


static void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

static inline unsigned int elementSizeTrt(DataType t)
{
	switch (t)
	{
	case DataType::kINT32:
		// Fallthrough, same as kFLOAT
	case DataType::kFLOAT: return 4;
	case DataType::kHALF: return 2;
	case DataType::kINT8: return 1;
	}
	assert(0);
	return 0;
}


static std::vector<std::pair<int64_t, DataType>>
calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<int64_t, DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }

    return sizes;
}

static ICudaEngine* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
                                      IUffParser* parser)
{
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

#if FLOAT_TYPE == USE_32BIT_FLOAT
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
#else // 16BIT_FLOAT
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kHALF))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
    builder->setFp16Mode(true);
#endif

    /* we create the engine */
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(MAX_WORKSPACE);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine)
        RETURN_AND_LOG(nullptr, ERROR, "Unable to create engine");

    /* we can clean the network and the parser */
    network->destroy();
    builder->destroy();

    return engine;
}

static void* createImageCudaBuffer(int64_t eltCount, DataType dtype, float *input_img)
{
  /* in that specific case, eltCount == INPUT_H * INPUT_W */
  assert(eltCount == INPUT_H * INPUT_W);
  assert(elementSizeTrt(dtype) == sizeof(float));

  size_t memSize = eltCount * elementSizeTrt(dtype);


  /* Allocate GPU memory and copy the image into it. */
  void* deviceMem = safeCudaMalloc(memSize);
  CHECK(cudaMemcpy(deviceMem, input_img, memSize, cudaMemcpyHostToDevice));

  return deviceMem;
}


static float getOutputs(int64_t eltCount, DataType dtype, void* buffer, int output_idx)
{
  /* This function copies a single output binding from GPU to memory. */
  float ret;
  
  LOG("%d eltCount\n", eltCount);
  assert(eltCount == OUTPUT_ELT_SZ[output_idx - NUM_OF_INPUTS]);  // FIXME this should validate the number of output elements. not sure if inputs are always first.
  assert(elementSizeTrt(dtype) == sizeof(float));
  LOG("--- OUTPUT %d ---\n", (output_idx - NUM_OF_INPUTS));

  size_t memSize = eltCount * elementSizeTrt(dtype);
  float* outputs = new float[eltCount];
  CHECK(cudaMemcpy(outputs, buffer, memSize, cudaMemcpyDeviceToHost));

  int maxIdx = 0;
  for (int i = 0; i < eltCount; ++i)
      if (outputs[i] > outputs[maxIdx])
          maxIdx = i;

#if MORE_THAN_1_ELEM_PER_OUTPUT
  /*
   * We should use this loop if there is more than 1 element per output.
   * Currently, this is not the usecase, so just stick in a varaible and return it
   */
  for (int64_t eltIdx = 0; eltIdx < eltCount; ++eltIdx)
  {
    LOG("%d => %d\t : ", eltIdx, outputs[eltIdx]);
    if (eltIdx == maxIdx)
      LOG("***");
    LOG("\n");
  }
#else
  // Expecting only a single element per output
  LOG("0 => %f\n", outputs[0]);
  ret = outputs[0];
#endif

  LOG("\n");
  delete[] outputs;
  return ret;
}


int inference(void* p_engine, void* p_context, float *input_img, float output_arr[NUM_OF_OUTPUTS])
{
  /* 
   * Get an image buffer ready for inference and run the NN on it.
   * The image is expected to be AFTER all preprocessing steps -
   *  croping, resizing, rescale and normalization (unless this is done by batchnorm).
   */
  LOG("TRTLib: clearing output array\n");
  memset(output_arr, 0, (sizeof(float) * NUM_OF_OUTPUTS));
  
  LOG("TRTLib: assigning from input pointers\n");
  
  ICudaEngine &engine = *((ICudaEngine*)p_engine);
  IExecutionContext* context = (IExecutionContext*)p_context;
  

  LOG("TRTLib: getting bindings from engine\n");
  int batchSize = 1;

  int nbBindings = engine.getNbBindings();
  assert(nbBindings == TOTAL_BINDINGS);

  std::vector<void*> buffers(nbBindings);
  auto buffersSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);
  
#ifdef DEBUG_IMAGE_VALUES
  LOG("Some sample values from the input array:  ");
  LOG("%f, %f, %f, %f\n", *input_img, *(input_img+20), *(input_img+30), *(input_img+50));
  LOG("%f, %f, %f, %f\n", *(input_img+39950), *(input_img+39970), *(input_img+39980), *(input_img+39999));

  LOG("TRTLib: assigning bindings\n");
#endif //DEBUG_IMAGE_VALUES

  int bindingIdxInput = 0;
  for (int i = 0; i < nbBindings; ++i)
  {
    if (engine.bindingIsInput(i))
    {
      bindingIdxInput = i;
    }
    else
    {
      auto bufferSizesOutput = buffersSizes[i];
      buffers[i] = safeCudaMalloc(bufferSizesOutput.first *
                                  elementSizeTrt(bufferSizesOutput.second));
    }
  }

  auto bufferSizesInput = buffersSizes[bindingIdxInput];

  LOG("TRTLib: creating buffer for input \n");

  buffers[bindingIdxInput] = createImageCudaBuffer(bufferSizesInput.first,
                                                   bufferSizesInput.second, input_img);

  LOG("TRTLib: executing inference\n");

  //auto t_start = std::chrono::high_resolution_clock::now();
  context->execute(batchSize, &buffers[0]);
  //auto t_end = std::chrono::high_resolution_clock::now();
  //ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
  //total += ms;

  LOG("TRTLib: moving output from GPU to host\n");

  int output_idx = 0;
  for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
  {
    float output;
    
    if (engine.bindingIsInput(bindingIdx))
      continue;

    auto bufferSizesOutput = buffersSizes[bindingIdx];
    output = getOutputs(bufferSizesOutput.first, bufferSizesOutput.second,
                        buffers[bindingIdx], bindingIdx);
    
    LOG("assigning output %f in array slot %d\n", output, output_idx);
    output_arr[output_idx++] = output;
  }
  
  LOG("TRTLib: clean GPU mem\n");
  
  // FIXME why not just clean all bindings? why separate input/output?
  CHECK(cudaFree(buffers[bindingIdxInput]));

  for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
    if (!engine.bindingIsInput(bindingIdx))
      CHECK(cudaFree(buffers[bindingIdx]));
      
  
  LOG("TRTLib: DONE\n");
  
  return 0;
}

int build_engine(std::string uff_path, uint8_t input_shape[2], void** out_engine, void** out_context)
{
  /*
   * This function will prepare a tensorRT engine, ready for inference jobs.
   * It should be called only once per NN.
   * 
   * @uff_path    : Full path to .uff model file.
   *                Note that this is not completely flexible, as input/output
   *                   size/names are hardcoded in the 'trtinference.h' file.
   * @input_shape : Integer array for input image size. should be [Height, Width].
   *                Only grayscale images (single channel) are supported now.
   */
  *out_engine = NULL;
  *out_context = NULL;
  
  LOG("TRTlib: %s\n", uff_path.c_str());
  LOG("TRTlib: %u,%u\n", input_shape[0], input_shape[1]);

  int maxBatchSize = 1;
  auto parser = createUffParser();

  INPUT_H = input_shape[0];
  INPUT_W = input_shape[1];

  /* Register tensorflow input */
  parser->registerInput(INPUT_BINDING_NAME,
                        Dims3(INPUT_C, INPUT_H, INPUT_W),
                        UffInputOrder::kNCHW);
  parser->registerOutput(OUTPUT_1_BINDING_NAME);
  parser->registerOutput(OUTPUT_2_BINDING_NAME);

  ICudaEngine* engine = loadModelAndCreateEngine(uff_path.c_str(), maxBatchSize, parser);

  if (!engine) {
    std::cout << "Failed to create engine" << std::endl;
    return -1;
  }

  /* we dont need to keep the memory created by the parser */
  parser->destroy();
  
  IExecutionContext* context = engine->createExecutionContext();

  *out_engine = (void*)engine;
  *out_context = (void*)context;

  return 0;
}

void delete_engine(void *p_engine_arg, void *p_context)
{
  LOG("TRTLib: clearing context\n");
  
  IExecutionContext* context = (IExecutionContext*)p_context;
  context->destroy();
  
  LOG("TRTLib: clearing engine\n");
  
  ICudaEngine* engine = (ICudaEngine*)p_engine_arg;
  
  engine->destroy();
  shutdownProtobufLibrary();
}

} // namespace trt_ros_inference
