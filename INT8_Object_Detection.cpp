#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

using namespace nvinfer1;

class YOLOCalibrator : public IInt8EntropyCalibrator2 {}
