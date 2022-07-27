#pragma once
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define FILE_READ "/home/linuxu/ParallelComputationFinalProject/input.txt" // change input text, if needed
#define MASTER 0
#define SLAVE 1

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

typedef struct match
{
	int isMatch;
	int objectId;
	int row;
	int col;
} Match;