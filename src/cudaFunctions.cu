#include "cudaFunctions.h"
#include <math.h>

__host__ void checkStatus(cudaError_t cudaStatus, const char* errorMsg)
{
    if(cudaStatus != cudaSuccess)
    {
        printf("%s\n",errorMsg);
        exit(1);
    }
}

__device__ int calcDiff(int p, int o)
{
    return abs((p - o) / p);
}

__global__ void findMatch(int* picture, int* object, int matchingValue, int picSize, int objSize, int* isMatch)
{
    int result = 0;
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int s = bx * blockDim.x + tx;
    int row = s / picSize;
    int col = s - picSize * row;
    if ((row + objSize) < picSize &&
        (col + objSize) < picSize)
    {
            for(int i = 0; i < objSize; i++)
            {
                for(int j = 0; j < objSize; j++)
                {
                    int objIdx = (i * objSize) + j;
                    int picIdx = ((row + i) * picSize) + (col + j);
                    result += calcDiff(picture[picIdx], object[objIdx]);
                    if (result > matchingValue)
                    {
                        *isMatch = 0;
                        return;
                    }
                }
            }
            *isMatch = 1;
    }
    else
    {
        *isMatch = 0;
    }
}


void cudaFuncs(Picture* picture, Obj* object, int* matchingValue)
{
    int *dev_pic = 0;
    int *dev_obj = 0;
    cudaError_t status = cudaSuccess;
    int numOfThreads, numOfBlocks;
    int pictureSize = picture->picSize;
    int objectSize = object->objSize;
    int* isMatch = 0;

    if ((pictureSize * pictureSize) > MAX_THREADS_IN_BLOCK)
    {
        numOfThreads = MAX_THREADS_IN_BLOCK;
        numOfBlocks = ((pictureSize * pictureSize)/numOfThreads) + 1;
    }
    else
    {
        numOfThreads = (pictureSize * pictureSize);
        numOfBlocks = 1;
    }

    // picture's device
    status = cudaMalloc((void**)&dev_pic, sizeof(int) * pictureSize * pictureSize);
    checkStatus(status, "Failded to allocate memory for picture in GPU\n");

    status = cudaMemcpy(dev_pic, picture->picArr, pictureSize*pictureSize*sizeof(int),cudaMemcpyHostToDevice);
    checkStatus(status, "CudaMemcpy to device failed! (dev_pic)\n");

    // object's device
    status = cudaMalloc((void**)&dev_obj, sizeof(int) * objectSize * objectSize);
    checkStatus(status, "Failded to allocate memory for object in GPU\n");

    status = cudaMemcpy(dev_obj, object->objArr, objectSize*objectSize*sizeof(int),cudaMemcpyHostToDevice);
    checkStatus(status, "CudaMemcpy to DEVICE Failed! (dev_obj)\n");


    // starting CUDA
    for(int row = 0; row < objectSize; row++)
    {
        for(int col = 0; col < objectSize; col++)
        {
            findMatch<<<numOfBlocks, numOfThreads>>>(dev_pic, dev_obj, *matchingValue, row, col, pictureSize, objectSize, isMatch);
        }
    }
    status = cudaDeviceSynchronize();
    checkStatus(status, "Synchronize Failed!\n");

    // //---------------- COPY DATA BACK TO HOST -----------------------
    // status = cudaMemcpy(bestMutant, dev_bestMutant, sizeof(Score),cudaMemcpyDeviceToHost);
    // checkStatus(status , "CudaMemcpy to device failed! (bestMutant)");

    // free memory
    status = cudaFree(dev_pic);
    checkStatus(status,"Cuda Free function failed! (dev_pic)\n");
    status = cudaFree(dev_obj);
    checkStatus(status,"Cuda Free function failed! (dev_obj)\n");
}