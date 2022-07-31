#pragma once
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "structsAndDefines.h"
#include "cudaFunctions.h"
#include "stddef.h"


/* This function reads data from input.txt */
void parseFile(char* path, float* matchingValue, int* numOfPics, Picture** pictures, int* numOfObjs, Obj** objects);

/* This function free pictures struct allocation from memory */
void freePictures(Picture** pictures, int numOfPics);

/* This function free objects struct allocation from memory */
void freeObjects(Obj** objects, int numOfObjs);

/* This function runs only from master process. Calculate portion to each process and send relevant data to specific process */
void runMaster(int p, char* path, Picture** pictures, Obj** objects, float* matching, int* numOfPics, int* numOfSlavePics, int* numOfObjs, Match** matches);

/* This function runs only from slaves process. Receiving data to work with */
void runSlave(Picture** pictures, Obj** objects, float* matching, int* numOfPics, int* numOfObjs, Match** matches);

/* This function print match result, otherwise, prints an appropriate message */
void printMatch(Match* myMatch);

/* This function get rank of process.
 * if this is a Slave process, then Slave process will send his result to Master process
 * if this is a Master process, then recieve results from Slave process and prints them.
 */
void printSlaveResult(Match* matches, int my_rank, int numOfSlavesPics);

/* This function search for match between objects and pictures */
void searchForMatch(Picture** pictures, Obj** objects, float* matching, int* numOfPics, int* numOfObjs, int my_rank, Match** matches);