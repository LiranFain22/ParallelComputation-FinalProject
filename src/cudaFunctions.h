#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"
#include "helper_functions.h"
#include "functions.h"

void findMatch(Picture** pictures, Obj** objects, int* matching, int* numOfPics, int* numOfObjs);