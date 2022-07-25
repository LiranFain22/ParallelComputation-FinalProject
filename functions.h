#pragma once
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "structsAndDefines.h"
#include "cudaFunctions.h"

/* This function reads data from input.txt */
void parseFile(char* path, float* matchingValue, int* numOfPics, Picture** pictures, int* numOfObjs, Obj** objects);

/* This function free pictures struct allocation from memory */
void freePictures(Picture** pictures, int numOfPics);

/* This function free objects struct allocation from memory */
void freeObjects(Obj** objects, int numOfObjs);

/* This function runs only from master process. Calculate portion to each process and send relevant data to specific process */
void runMaster(int p, char* path, Picture** pictures, Obj** objects, float* matching, int* numOfPics, int* numOfObjs);

/* This function runs only from slaves process. Receiving data to work with */
void runSlave(Picture** pictures, Obj** objects, float* matching, int* numOfPics, int* numOfObjs);

/* This function search for match between objects and pictures */
void searchForMatch(Picture** pictures, Obj** objects, float* matching, int* numOfPics, int* numOfObjs);