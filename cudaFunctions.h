#pragma once
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "structs.h"
#include "functions.h"

#define MAX_THREADS_IN_BLOCK 1024

// __device__ int calcDiff(int p, int o);

// __global__ void findMatch(int* picture, int* object, int matchingValue, int picSize, int objSize, int* isMatch);

void cudaFuncs(Picture* picture, Obj* object, int* matchingValue);