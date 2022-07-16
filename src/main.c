#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "mpi.h"

void parseFile(char* path, int* matchingValue, int* numOfPics, int*** pictures, int* numOfObjs, int*** objects)
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

	printf("matchingValue: %d\nnumOfPics: %d\n", *matchingValue, *numOfPics);

	// Creating size of pictures
	*(pictures) = (int**)malloc(sizeof(int*)*(*numOfPics));
	if(!*(pictures))
	{
		printf("Failed to create pictures..\n");
		exit(1);
	}

	// Inserting numbers to each picture
	for(int i = 0; i < *(numOfPics); i++)
	{
		int indexPic;
		int sizeOfEachPic;
		fscanf(fp, "%d %d", &indexPic, &sizeOfEachPic);
		(*pictures)[indexPic - 1] = (int*)malloc(sizeof(int)*sizeOfEachPic*sizeOfEachPic);
		for(int j = 0; j < sizeOfEachPic*sizeOfEachPic; j++)
		{
			fscanf(fp, "%d", &(*pictures)[i][j]);
		}
	}

	if(fscanf(fp, "%d", numOfObjs) == EOF)
	{
		printf("Could not read numOfObjs from file!\n");fflush(stdout);
		exit(1);
	}

	printf("nnumOfObjs: %d\n", *numOfObjs);

	*(objects) = (int**)malloc(sizeof(int*)*(*numOfObjs));
	if(!*(objects))
	{
		printf("Failed to create objects..\n");
		exit(1);
	}

	// Inseting numbers to each object
	for(int i = 0; i < *(numOfObjs); i++)
	{
		int indexObj;
		int sizeOfEachObj;
		fscanf(fp, "%d %d", &indexObj, &sizeOfEachObj);
		(*objects)[indexObj - 1] = (int*)malloc(sizeof(int)*sizeOfEachObj*sizeOfEachObj);
		for(int j = 0; j < sizeOfEachObj*sizeOfEachObj; j++)
		{
			fscanf(fp, "%d", &(*objects)[i][j]);
		}
	}
}

void runMaster(int p, char* path, int ***myPictures)
{
	int matching;
	int numOfPics;
	int** pictures;
	int numOfObjs;
	int** objects;

	parseFile(path, &matching, &numOfPics, &pictures, &numOfObjs, &objects);


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
		}
	}
}

void runSlave()
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

	// TODO: implement malloc for objects array and each object in the array

	MPI_Recv(&objects, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	printf("Objects array from send/recv start:\n");
	for(int i = 0; i < numOfObjs; i++)
	{
		printf("%d\n", *objects[i]);
	}
	printf("Objects array from send/recv end\n");
	MPI_Recv(&portionSize, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	printf("portionSize from send/recv = %d\n", portionSize);
	printf("pictures array from send/recv start:\n");
	for(int i = 0; i < portionSize; i++)
	{
		MPI_Recv(pictures[i], 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("%d\n", pictures[i]);
	}
	printf("pictures array from send/recv end\n");
}

int main(int argc, char* argv[]){
	int  my_rank; /* rank of process */
	int  p;       /* number of processes */
	int source;   /* rank of sender */
	int dest;     /* rank of receiver */
	int tag=0;    /* tag for messages */
	char message[100];        /* storage for message */
	MPI_Status status ;   /* return status for receive */

	/* start up MPI */

	MPI_Init(&argc, &argv);

	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &p);


	if (my_rank !=0){
		runSlave();
	}
	else{
		int** myPictures;
		runMaster(p, "/home/linuxu/ParallelComputationFinalProject/src/input2.txt", &myPictures);
	}
	/* shut down MPI */
	MPI_Finalize();


	
	return 0;
}
