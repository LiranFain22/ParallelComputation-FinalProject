#include "cudaFunctions.h"
#include <math.h>

__device__ int calcDiff(int p, int o)
{
    return abs((p - o) / p);
}

__global__ void findMatch(Picture* picture, Obj* object, int matchingValue, int row, int col, int picSize, int objSize, int* isMatch)
{
    int result = 0;
    if ((row + object->objSize) < picture->picSize &&
        (col + object->objSize) < picture->picSize)
    {
            for(int i = 0; i < object->objSize; i++)
            {
                for(int j = 0; j < object->objSize; j++)
                {
                    int objIdx = (i * object->objSize) + j;
                    int picIdx = ((row + i) * picture->picSize) + (col + j);
                    result += calcDiff(picture->picArr[picIdx], object->objArr[objIdx]);
                    if ( result > matchingValue)
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
    Picture *dev_pic = 0;
    Obj *dev_object = 0;
    cudaError_t status = cudaSuccess;
    int numOfThreads, numOfBlocks;
    int pictureSize = picture->picSize;
    int objectSize = object->objSize;

    if ((picture->picSize * picture->picSize) > MAX_THREADS_IN_BLOCK)
    {
        numOfThreads = MAX_THREADS_IN_BLOCK;
        numOfBlocks = ((picture->picSize * picture->picSize)/numOfThreads) + 1;
    }
    else
    {
        numOfThreads = (picture->picSize * picture->picSize);
        numOfBlocks = 1;
    }

    status = cudaMalloc((void**)&dev_pic, sizeof(int) * pictureSize);
    if(status != cudaSuccess)
    {
        printf("Failded to allocate memory for picture in GPU\n");
        exit(1);
    }
    status = cudaMemcpy(dev_seq1, seq1, seq1Len*sizeof(char),cudaMemcpyHostToDevice);
    checkStatus(status , "CudaMemcpy to DEVICE Failed! (dev_seq1)");

    //Seq2
    status = cudaMalloc((void**)&dev_seq2, sizeof(char) * seq2Len);
    checkStatus(status , "CudaMalloc Failed! (dev_seq2)");
    status = cudaMemcpy(dev_seq2, seq2, seq2Len*sizeof(char),cudaMemcpyHostToDevice);
    checkStatus(status , "CudaMemcpy to DEVICE Failed! (dev_seq2)");

    //Weight array
    status = cudaMalloc((void**)&dev_weights, sizeof(float) * WEIGHTS);
    checkStatus(status , "CudaMalloc Failed! (dev_weights)");
    status = cudaMemcpy(dev_weights, weightArr, WEIGHTS*sizeof(float),cudaMemcpyHostToDevice);
    checkStatus(status , "CudaMemcpy to DEVICE Failed! (dev_weights)");

    //Device best score
    status = cudaMalloc((void**)&dev_bestMutant, sizeof(Score));
    checkStatus(status , "CudaMalloc Failed! (dev_bestMutant)tell me why??");
    status = cudaMemcpy(dev_bestMutant, &tempScore, sizeof(Score),cudaMemcpyHostToDevice);
    checkStatus(status , "CudaMemcpy to DEVICE Failed! (dev_weights)");

    //Device score array
    status = cudaMalloc((void**)&dev_mutantArr, sizeof(Score) * (*mutantArrSize));
    checkStatus(status , "CudaMalloc Failed! (dev_bestMutant)");
    status = cudaMemcpy(dev_mutantArr, mutantArr, sizeof(Score) * (*mutantArrSize),cudaMemcpyHostToDevice);
    checkStatus(status , "CudaMemcpy to DEVICE Failed! (dev_weights)");

    //---------------- START CUDA -----------------------
    cudaCalculations<<<numOfBlocks, numOfThreads>>>(dev_seq1, dev_seq2, seq2Len, dev_mutantArr, *mutantArrSize, dev_weights, dev_bestMutant);
    status = cudaDeviceSynchronize();
    checkStatus(status , "Synchronize Failed!");

    //---------------- COPY DATA BACK TO HOST -----------------------
    status = cudaMemcpy(bestMutant, dev_bestMutant, sizeof(Score),cudaMemcpyDeviceToHost);
    checkStatus(status , "CudaMemcpy to DEVICE Failed! (bestMutant)");

    //---------------- FREE MEMORY -----------------------
    status = cudaFree(dev_seq1);
    checkStatus(status,"Cuda Free Failed! (dev_seq1)");
    status = cudaFree(dev_seq2);
    checkStatus(status,"Cuda Free Failed! (dev_seq2)");
    status = cudaFree(dev_weights);
    checkStatus(status,"Cuda Free Failed! (dev_weights)");
    status = cudaFree(dev_bestMutant);
    checkStatus(status,"Cuda Free Failed! (dev_bestMutant)");
    status = cudaFree(dev_mutantArr);
    checkStatus(status,"Cuda Free Failed! (dev_mutantArr)");
}