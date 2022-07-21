#pragma once
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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