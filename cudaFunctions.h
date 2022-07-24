#pragma once
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "structs.h"
#include "functions.h"

#define MAX_THREADS_IN_BLOCK 1024

/* This function calculate match with CUDA functions */
void cudaFuncs(Picture* picture, Obj* object, float* matchingValue, Match* match);
