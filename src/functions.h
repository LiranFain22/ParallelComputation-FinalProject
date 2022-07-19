#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

typedef struct picture
{
	int picId;
	int picSize;
	int* picArr;
} Picture;

typedef struct obj
{
	int objId;
	int objSize;
	int* objArr;
} Obj;


void parseFile(char* path, int* matchingValue, int* numOfPics, Picture** pictures, int* numOfObjs, Obj** objects);

void freePictures(Picture** pictures, int numOfPics);

void freeObjects(Obj** objects, int numOfObjs);

void runMaster(int p, char* path, Picture** pictures, Obj** objects, int* matching, int* numOfPics, int* numOfObjs);

void runSlave(Picture** pictures, Obj** objects, int* matching, int* numOfPics, int* numOfObjs);

void findMatch(Picture** pictures, Obj** objects, int* matching, int* numOfPics, int* numOfObjs);