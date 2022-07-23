#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h" 
#include "helper_functions.h" 
#include "cudaFunctions.h"
#include <math.h>

void checkStatus(cudaError_t cudaStatus, const char* errorMsg)
{
    if(cudaStatus != cudaSuccess)
    {
        perror(errorMsg);
        exit(1);
    }
}

__device__ float calcDiff(float p, float o)
{
    return abs((p - o) / p);
}

// __device__ void printArray(int* arr, int arrSize, int threadNum, int isPic)
// {
//     if(isPic == 1)
//         printf("----picture start for thread num: %d ----\n", threadNum);
//     else
//         printf("----object start for thread num: %d ----\n", threadNum);

//     for(int i=0; i<arrSize; i++)
//     {
//         for(int j=0; j<arrSize; j++)
//         {
//             printf("|%d|", arr[i*arrSize + j]);
//         }
//         printf("\n");
//     }
//     printf("---- array end ----\n\n");
// }

__global__ void findMatch(int* picture, int* object, float matchingValue, int picSize, int objSize, Match* match, int objectId)
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
                    if(row == 1 && col == 1)
                    {
                        printf("picture[picIdx] = %d, object[objIdx] = %d, result = %d\n",  picture[picIdx], object[objIdx], result);
                    }
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
                printf("cudaaaaaa found match in row = %d and col = %d\n", row, col);
            }
            else
            {
                (*match).isMatch = 0;
            }
    }

}


void cudaFuncs(Picture* picture, Obj* object, float* matchingValue, Match* match)
{
    int *dev_pic = 0;
    int *dev_obj = 0;
    cudaError_t status = cudaSuccess;
    int numOfThreads, numOfBlocks;
    int pictureSize = picture->picSize;
    int objectSize = object->objSize;
    int objId = object->objId;
    Match* dev_match = 0;
    // dev_match->isMatch = 0;

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

    // printf("starting cude with match is match = %d\n", (*match).isMatch);
    // printf("picSize for allocation is %d\n", pictureSize);
    // picture's device
    status = cudaMalloc((void**)&dev_pic, sizeof(int) * pictureSize * pictureSize);
    checkStatus(status, "Faild to allocate memory for picture in GPU\n");

    // printf("succeeded allocating memory for dev_pic\n");

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
    findMatch<<<numOfBlocks, numOfThreads>>>(dev_pic, dev_obj, *matchingValue, pictureSize, objectSize, dev_match, objId);
    
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