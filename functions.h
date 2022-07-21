#pragma once
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "structs.h"
#include "cudaFunctions.h"


void parseFile(char* path, int* matchingValue, int* numOfPics, Picture** pictures, int* numOfObjs, Obj** objects);

void freePictures(Picture** pictures, int numOfPics);

void freeObjects(Obj** objects, int numOfObjs);

void runMaster(int p, char* path, Picture** pictures, Obj** objects, int* matching, int* numOfPics, int* numOfObjs);

void runSlave(Picture** pictures, Obj** objects, int* matching, int* numOfPics, int* numOfObjs);

void searchForMatch(Picture** pictures, Obj** objects, int* matching, int* numOfPics, int* numOfObjs, int my_rank);