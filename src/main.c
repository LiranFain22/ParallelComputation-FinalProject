#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "mpi.h"

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

void parseFileOrigin(char* path, int* matchingValue, int* numOfPics, Picture** pictures, int* numOfObjs, Obj** objects)
{
	FILE* fp;

	if((fp = fopen(path, "r")) == 0)
	{
		printf("Could not open file\n");
		exit(1);
	}

	if(fscanf(fp, "%d %d", matchingValue, numOfPics) == EOF)
	{
		printf("Could not read matching and numOfPics from file!\n");fflush(stdout);
		exit(1);
	}
	// TODO delete printf
	printf("matchingValue: %d\n\nnumOfPics: %d\n\n", *matchingValue, *numOfPics);

	// Creating size of pictures
	*(pictures) = (Picture*)malloc(sizeof(Picture)*(*numOfPics));
	if(!*(pictures))
	{
		printf("Failed to create pictures..\n");
		exit(1);
	}

	// Inserting numbers to each picture
	for(int i = 0; i < *(numOfPics); i++)
	{
		int indexPic;
		fscanf(fp, "%d", &indexPic);
		pictures[i]->picId = indexPic - 1; // base 1 NOT base 0
		// TODO delete printf
		printf("Picture index: %d\n", pictures[i]->picId);
		// Reads from file each picture size
		fscanf(fp, "%d", &pictures[i]->picSize);
		// TODO delete printf
		printf("Picture size: %d\n\n", pictures[i]->picSize);

		(*pictures)[i].picArr = (int*)malloc(sizeof(int)*pictures[i]->picSize*pictures[i]->picSize);
		for(int j = 0; j < pictures[i]->picSize * pictures[i]->picSize; j++)
		{
			fscanf(fp, "%d", &(*pictures)[i].picArr[j]);
			// TODO delete printf
			printf("Picture[%d][%d] = %d\n", i, j, (*pictures)[i].picArr[j]);
		}
		// TODO delete printf
		printf("\n\n");
	}

	if(fscanf(fp, "%d", numOfObjs) == EOF)
	{
		printf("Could not read numOfObjs from file!\n");fflush(stdout);
		exit(1);
	}
	// TODO delete printf
	printf("numOfObjs: %d\n\n", *numOfObjs);

	// Creating size of objects
	*(objects) = (Obj*)malloc(sizeof(Obj)*(*numOfObjs));
	if(!*(objects))
	{
		printf("Failed to create objects..\n");
		exit(1);
	}

	// Inserting numbers to each object
	for(int  i = 0; i < *(numOfObjs); i++)
	{
		int indexObj;
		fscanf(fp, "%d", &indexObj);
		objects[i]->objId = indexObj - 1; // base 1 NOT base 0
		// TODO delete printf
		printf("Object index: %d\n", objects[i]->objId);

		// Read from file each object size
		fscanf(fp, "%d", &objects[i]->objSize);
		// TODO delete printf
		printf("Object size: %d\n\n", objects[i]->objSize);

		(*objects)[i].objArr = (int*)malloc(sizeof(int)*objects[i]->objSize*objects[i]->objSize);
		for(int j = 0; j < objects[i]->objSize * objects[i]->objSize; j++)
		{
			fscanf(fp, "%d", &(*objects)[i].objArr[j]);
			// TODO delete printf
			printf("Object[%d][%d] = %d\n", i, j, (*objects)[i].objArr[j]);
		}
		// TODO delete printf
		printf("\n\n");
	}
}

/* void parseFile(char* path, int* matchingValue, int* numOfPics, int*** pictures, int* numOfObjs, int*** objects, int** arrOfEachPic, int** arrOfEachObj)
{
	FILE* fp;

	if((fp = fopen(path, "r")) == 0)
	{
		printf("Could not open file\n");
		exit(1);
	}

	if(fscanf(fp, "%d %d", matchingValue, numOfPics) == EOF)
	{
		printf("Could not read matching and numOfPics from file!\n");fflush(stdout);
		exit(1);
	}

	printf("matchingValue: %d\nnumOfPics: %d\n\n", *matchingValue, *numOfPics);

	// Creating size of pictures
	*(pictures) = (int**)malloc(sizeof(int*)*(*numOfPics));
	if(!*(pictures))
	{
		printf("Failed to create pictures..\n");
		exit(1);
	}

	// Creating array of sizes of each picture
	*(arrOfEachPic) = (int*)malloc(sizeof(int)*(*numOfPics));
	if(!arrOfEachPic)
	{
		printf("Failed to create array of sizes of each picture..\n");
		exit(1);
	}

	// Inserting numbers to each picture
	for(int i = 0; i < *(numOfPics); i++)
	{
		int indexPic;
		int sizeOfEachPic;
		fscanf(fp, "%d %d", &indexPic, &sizeOfEachPic);
		// delete print
		// printf("index pic = %d\nsizeOfEachPic = %d\n", (indexPic - 1), sizeOfEachPic);
		(*arrOfEachPic)[indexPic - 1] = sizeOfEachPic;
		// delete print
		printf("arrOfEachPic[%d] = %d\n\n", (indexPic - 1), (*arrOfEachPic)[indexPic - 1]);

		(*pictures)[indexPic - 1] = (int*)malloc(sizeof(int)*sizeOfEachPic*sizeOfEachPic);
		for(int j = 0; j < sizeOfEachPic*sizeOfEachPic; j++)
		{
			fscanf(fp, "%d", &(*pictures)[i][j]);
			// delete print
			printf("pictures[%d][%d] = %d\n", i, j, (*pictures)[i][j]);
		}
		printf("\n");
	}


	if(fscanf(fp, "%d", numOfObjs) == EOF)
	{
		printf("Could not read numOfObjs from file!\n");fflush(stdout);
		exit(1);
	}

	printf("numOfObjs: %d\n\n", *numOfObjs);

	// Creating size of objects
	*(objects) = (int**)malloc(sizeof(int*)*(*numOfObjs));
	if(!*(objects))
	{
		printf("Failed to create objects..\n");
		exit(1);
	}

	// Creating array of sizes of each object
	*(arrOfEachObj) = (int*)malloc(sizeof(int)*(*numOfObjs));
	if(!arrOfEachObj)
	{
		printf("Failed to create array of sizes of each object..\n");
		exit(1);
	}	

	// Inseting numbers to each object
	for(int i = 0; i < *(numOfObjs); i++)
	{
		int indexObj;
		int sizeOfEachObj;
		fscanf(fp, "%d %d", &indexObj, &sizeOfEachObj);
		// delete print
		// printf("index obj = %d\nsizeOfEachObj = %d\n", (indexObj - 1), sizeOfEachObj);
		(*arrOfEachObj)[indexObj - 1] = sizeOfEachObj;
		// delete print
		printf("arrOfEachObj[%d] = %d\n\n", (indexObj - 1), (*arrOfEachObj)[indexObj - 1]);

		(*objects)[indexObj - 1] = (int*)malloc(sizeof(int)*sizeOfEachObj*sizeOfEachObj);
		for(int j = 0; j < sizeOfEachObj*sizeOfEachObj; j++)
		{
			fscanf(fp, "%d", &(*objects)[i][j]);
			// delete print
			printf("objects[%d][%d] = %d\n", i, j, (*objects)[i][j]);
		}
		printf("\n");
	}
} */

void runMasterOrigin(int p, char* path, Picture** pictures, Obj** objects, int* matching, int* numOfPics, int* numOfObjs)
{
	parseFileOrigin(path, matching, numOfPics, pictures, numOfObjs, objects);

	// TODO delete printf
	printf("YAY!\n");

	if (p > (*numOfPics))
	{
		p = (*numOfPics);
	}

	int portionSize = (*numOfPics) / p;

	for(int i = 0; i < p; i++)
	{
		if (i == 0)
		{
			for(int j = 0; j < portionSize; j++)
			{
				// TODO: implement matser's part - pictures
			}
		} else
		{
			MPI_Send(matching, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(numOfObjs, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(objects, 1, sizeof(Obj), i, 0, MPI_COMM_WORLD);
			MPI_Send(&portionSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			// TODO: implement slave's part - pictures
			// TODO: implement slave's part - objects
		}
	}
}

/* void runMaster(int p, char* path, int ***myPictures)
{
	int matching;
	int numOfPics;
	int** pictures;
	int numOfObjs;
	int** objects;
	int* arrOfEachPic;
	int* arrOfEachObj;

	parseFile(path, &matching, &numOfPics, &pictures, &numOfObjs, &objects, &arrOfEachPic, &arrOfEachObj);


	if (p > numOfPics)
	{
		p = numOfPics;
	}

	int portionSize = numOfPics / p;

	*myPictures = (int**)malloc(portionSize * sizeof(int*));
	for(int i = 0; i < p; i++)
	{
		if(i==0){
			for(int j = 0; j < portionSize; j++)
			{
				(*myPictures)[j] = pictures[j];
			}
		}else{
			MPI_Send(&matching, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			printf("send matcing = %d\n", matching);

			MPI_Send(&numOfObjs, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			printf("send numOfObj = %d\n", numOfObjs);

			MPI_Send(objects, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			printf("send objects\n");
			
			MPI_Send(&portionSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			printf("send portionSize\n");

			for(int j = i * portionSize; j < i * (portionSize + 1); j++)
			{
				MPI_Send(pictures[j],1, MPI_INT, i, 0, MPI_COMM_WORLD);
			}
			printf("send pictures\n");

			MPI_Send(arrOfEachPic, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			printf("send arrOfEachPic\n");

			MPI_Send(arrOfEachObj, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			printf("send arrOfEachObj\n");
		}
	}
} */

void runSlaveOrigin(Picture** pictures, Obj** objects, int* matching, int* numOfPics, int* numOfObjs)
{
	MPI_Recv(&matching, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	MPI_Recv(&numOfObjs, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	*(objects) = (Obj*)malloc(sizeof(Obj)*(*numOfObjs));
	if(!*(objects))
	{
		printf("Failed to malloc objects..\n");
		exit(1);
	}


}

/* void runSlave()
{
	int matching;
	int numOfObjs;
	int** objects;
	int portionSize;
	int** pictures;

	MPI_Recv(&matching, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	printf("matching from send/recv = %d\n", matching);
	MPI_Recv(&numOfObjs, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	printf("numOfObjs from send/recv = %d\n", numOfObjs);

	objects = (int**)malloc(sizeof(int*)*numOfObjs);
	if(!objects)
	{
		printf("Failed to malloc objects..\n");
		exit(1);
	}
	MPI_Recv(&objects, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	// printf("Objects array from send/recv start:\n");
	// for(int i = 0; i < numOfObjs; i++)
	// {
	// 	printf("objects[%d] = %d\n", i, (*objects)[i]);
	// }
	// printf("Objects array from send/recv end\n");



	MPI_Recv(&portionSize, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	printf("portionSize from send/recv = %d\n", portionSize);

	printf("pictures array from send/recv start:\n");
	for(int i = 0; i < portionSize; i++)
	{
		MPI_Recv(pictures[i], 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("%d\n", (*pictures)[i]);
	}
	printf("pictures array from send/recv end\n");
} */

int main(int argc, char* argv[]){
	int  my_rank; /* rank of process */
	int  p;       /* number of processes */
	int matching;
	int numOfPics;
	int numOfObjs;
	Picture* pictures;
	Obj* objects;

	/* start up MPI */

	MPI_Init(&argc, &argv);

	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &p);


	if (my_rank !=0){
		runSlaveOrigin(&pictures, &objects, &matching, &numOfPics, &numOfObjs);
	}
	else{
		runMasterOrigin(p, "/home/linuxu/ParallelComputationFinalProject/src/input2.txt", &pictures, &objects, &matching, &numOfPics, &numOfObjs);
	}
	/* shut down MPI */
	MPI_Finalize();


	
	return 0;
}
