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

int main(int argc, char* argv[]){
//	int  my_rank; /* rank of process */
//	int  p;       /* number of processes */
//	int source;   /* rank of sender */
//	int dest;     /* rank of receiver */
//	int tag=0;    /* tag for messages */
//	char message[100];        /* storage for message */
//	MPI_Status status ;   /* return status for receive */
//
//	/* start up MPI */
//
//	MPI_Init(&argc, &argv);
//
//	/* find out process rank */
//	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
//
//	/* find out number of processes */
//	MPI_Comm_size(MPI_COMM_WORLD, &p);
//
//
//	if (my_rank !=0){
//		/* create message */
//		sprintf(message, "Hello MPI World from process %d!", my_rank);
//		dest = 0;
//		/* use strlen+1 so that '\0' get transmitted */
//		MPI_Send(message, strlen(message)+1, MPI_CHAR,
//		   dest, tag, MPI_COMM_WORLD);
//	}
//	else{
//		printf("Hello MPI World From process 0: Num processes: %d\n",p);
//		for (source = 1; source < p; source++) {
//			MPI_Recv(message, 100, MPI_CHAR, source, tag,
//			      MPI_COMM_WORLD, &status);
//			printf("%s\n",message);
//		}
//	}
//	/* shut down MPI */
//	MPI_Finalize();
//
//
	int matching;
	int numOfPics;
	int** pictures;
	int numOfObjs;
	int** objects;
	parseFile("/home/linuxu/ParallelComputationFinalProject/src/input2.txt", &matching, &numOfPics, &pictures, &numOfObjs, &objects);
	return 0;
}
