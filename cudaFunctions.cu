#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h" 
#include "helper_functions.h" 
#include "cudaFunctions.h"
#include <math.h>

/*
 * This function checks if a command was completed successfully, 
 * if not, sends an appropriate error message.
 */
void checkStatus(cudaError_t cudaStatus, const char* errorMsg)
{
    if(cudaStatus != cudaSuccess)
    {
        perror(errorMsg);
        exit(1);
    }
}

/*
 * This function calculate difference for overlapping members
 */
__device__ float calcDiff(float p, float o)
{
    return abs((p - o) / p);
}

/*
 * This function finds a match using calcDiff function.
 */
__global__ void findMatch(int* picture, int* object, float matchingValue, int picSize, int objSize, Match* match, int objectId, int picId)
{
    float result = 0.0;

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int s = bx * blockDim.x + tx;
    int foundMatch = 1;
    __shared__ int bestMatchIdx;
    bestMatchIdx = -1;

    int row = s / picSize;
    int col = s - picSize * row;


    if ((row + objSize) <= picSize && (col + objSize) <= picSize)
    {
            for(int i = 0; i < objSize; i++)
            {
                for(int j = 0; j < objSize; j++)
                {
                    int objIdx = (i * objSize) + j;
                    int picIdx = ((row + i) * picSize) + (col + j);
                    
                    result += calcDiff(__int2float_rd(picture[picIdx]), __int2float_rd(object[objIdx]));

                    if (result > matchingValue || match->isMatch==1)
                    {
                        foundMatch = 0;
                        break;
                    }
                }
                if(foundMatch == 0) break;
            }
            __syncthreads();
            //atomic min if foundMatch is 1
            if (foundMatch == 1) 
            {
                atomicMax(&bestMatchIdx, s);
            }
            __syncthreads();
            // check if i am min
                // if i do, update is match with row, col, obj id, is match
            if (s == bestMatchIdx)
            {
                (*match).isMatch = 1;
                (*match).row = row;
                (*match).col = col;
                (*match).objectId = objectId;
                (*match).picId = picId;
            }
            else
            {
                (*match).isMatch = 0;
            }
    }

}

/*
 * This function calculate match with CUDA functions
 */
void cudaFuncs(Picture* picture, Obj* object, float* matchingValue, Match* match)
{
    int *dev_pic = 0;
    int *dev_obj = 0;
    cudaError_t status = cudaSuccess;
    int numOfThreads, numOfBlocks;
    int pictureSize = picture->picSize;
    int objectSize = object->objSize;
    int objId = object->objId;
    int picId = picture->picId;
    Match* dev_match = 0;

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
    checkStatus(status, "Faild to allocate memory for picture in GPU\n");


    status = cudaMemcpy(dev_pic, picture->picArr, pictureSize*pictureSize*sizeof(int),cudaMemcpyHostToDevice);
    checkStatus(status, "CudaMemcpy to device failed! (dev_pic)\n");

    // object's device
    status = cudaMalloc((void**)&dev_obj, sizeof(int) * objectSize * objectSize);
    checkStatus(status, "Faild to allocate memory for object in GPU\n");

    status = cudaMemcpy(dev_obj, object->objArr, objectSize*objectSize*sizeof(int),cudaMemcpyHostToDevice);
    checkStatus(status, "CudaMemcpy to device failed! (dev_obj)\n");

    // match's device
    status = cudaMalloc((void**)&dev_match, sizeof(Match));
    checkStatus(status, "Faild to allocate memory for match in GPU\n");


    status = cudaMemcpy(dev_match, match, sizeof(Match),cudaMemcpyHostToDevice);
    checkStatus(status, "CudaMemcpy to device failed! (dev_match)\n");


    // starting CUDA
    findMatch<<<numOfBlocks, numOfThreads>>>(dev_pic, dev_obj, *matchingValue, pictureSize, objectSize, dev_match, objId, picId);
    
    status = cudaDeviceSynchronize();
    checkStatus(status, "Synchronize Failed!\n");

    // copy data back to host
    status = cudaMemcpy(match, dev_match, sizeof(Match),cudaMemcpyDeviceToHost);
    checkStatus(status, "CudaMemcpy to host failed! (isMatch)\n");

    // free memory
    status = cudaFree(dev_pic);
    checkStatus(status,"Cuda Free function failed! (dev_pic)\n");

    status = cudaFree(dev_obj);
    checkStatus(status,"Cuda Free function failed! (dev_obj)\n");

    status = cudaFree(dev_match);
    checkStatus(status,"Cuda Free function failed! (dev_match)\n");

}