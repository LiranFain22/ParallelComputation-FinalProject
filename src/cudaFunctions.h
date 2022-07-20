#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"
#include "helper_functions.h"

#define MAX_THREADS_IN_BLOCK 1024

__host__ void checkStatus(cudaError_t cudaStatus, const char* errorMsg);

__device__ int calcDiff(int p, int o);

__global__ void findMatch(int* picture, int* object, int matchingValue, int row, int col, int picSize, int objSize, int* isMatch);

void cudaFuncs(Picture* picture, Obj* object, int* matchingValue);