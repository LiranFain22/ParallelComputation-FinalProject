#include "functions.h"
#include "mpi.h"

void parseFile(char* path, int* matchingValue, int* numOfPics, Picture** pictures, int* numOfObjs, Obj** objects)
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
		fscanf(fp, "%d", &(*pictures)[i].picId);
		// TODO delete printf
		printf("Picture index: %d\n", (*pictures)[i].picId);
		// Reads from file each picture size
		fscanf(fp, "%d", &(*pictures)[i].picSize);
		// TODO delete printf
		printf("Picture size: %d\n\n", (*pictures)[i].picSize);

		(*pictures)[i].picArr = (int*)malloc(sizeof(int)*(*pictures)[i].picSize*(*pictures)[i].picSize);
		for(int j = 0; j < (*pictures)[i].picSize * (*pictures)[i].picSize; j++)
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
		fscanf(fp, "%d", &(*objects)[i].objId);
		// TODO delete printf
		printf("Object index: %d\n", (*objects)[i].objId);

		// Read from file each object size
		fscanf(fp, "%d", &(*objects)[i].objSize);
		// TODO delete printf
		printf("Object size: %d\n\n", (*objects)[i].objSize);

		(*objects)[i].objArr = (int*)malloc(sizeof(int)*(*objects)[i].objSize*(*objects)[i].objSize);
		for(int j = 0; j < (*objects)[i].objSize * (*objects)[i].objSize; j++)
		{
			fscanf(fp, "%d", &(*objects)[i].objArr[j]);
			// TODO delete printf
			printf("Object[%d][%d] = %d\n", i, j, (*objects)[i].objArr[j]);
		}
		// TODO delete printf
		printf("\n\n");
	}
}

void freePictures(Picture** pictures, int numOfPics)
{
    for(int i = 0; i < numOfPics; i++)
    {
		free((*pictures)[i].picArr);
    }
	free(pictures);
}

void freeObjects(Obj** objects, int numOfObjs)
{
    for(int i = 0; i < numOfObjs; i++)
    {
		free((*objects)[i].objArr);
    }
	free(objects);
}

void runMaster(int p, char* path, Picture** pictures, Obj** objects, int* matching, int* numOfPics, int* numOfObjs)
{
	Picture* allPictures;
	int numOfAllPictures;

	parseFile(path, matching, &numOfAllPictures, &allPictures, numOfObjs, objects);

	if (p > numOfAllPictures)
	{
		p = numOfAllPictures;
	}

	int portionSize = numOfAllPictures / p;
	*numOfPics = portionSize;

	for(int i = 0; i < p; i++)
	{
		if (i == 0)
		{
			*pictures = (Picture*)malloc(sizeof(Picture)*portionSize);
			if(!*pictures)
			{
				printf("Failed to create pictures..\n");
				exit(1);
			}
			for(int j = 0; j < portionSize; j++)
			{
				(*pictures)[j].picId = allPictures[j].picId;
				(*pictures)[j].picSize = allPictures[j].picSize;
				(*pictures)[j].picArr = (int*)malloc(sizeof(int)*(*pictures)[j].picSize);
				for(int k = 0; k < (*pictures)[j].picSize * (*pictures)[j].picSize; j++)
				{
					(*pictures)[j].picArr[k] = allPictures[j].picArr[k];
				}
			}
		} else
		{
			MPI_Send(matching, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(numOfObjs, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			for(int j = 0; j < (*numOfObjs); j++)
			{
				MPI_Send(&(*objects)[j].objId, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&(*objects)[j].objSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				for(int k = 0; k < (*objects)[j].objSize * (*objects)[j].objSize; k++)
				{
					MPI_Send(&(*objects)[j].objArr[k], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				}
			}
			MPI_Send(&portionSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			// TODO delete print
			printf("from master portionSize = %d\n", portionSize);
			for(int j = portionSize * i; j < portionSize * (i + 1) ; j++)
			{
				MPI_Send(&(allPictures)[j].picId, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				// TODO delete print
				printf("from master picture id = %d\n", (allPictures)[j].picId);
				printf("from master j = %d\n", j);

				MPI_Send(&(allPictures)[j].picSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				for(int k = 0; k < (allPictures)[j].picSize * (allPictures)[j].picSize; k++)
				{
					MPI_Send(&(allPictures)[j].picArr[k], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				}
			}
		}
	}
	freePictures(pictures, *numOfPics);
}

void runSlave(Picture** pictures, Obj** objects, int* matching, int* numOfPics, int* numOfObjs)
{
	// TODO delete print
	printf("\n\nIn Slave function:\n\n");

	MPI_Recv(matching, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	// TODO delete print
	printf("from slave matching = %d\n", *matching);

	MPI_Recv(numOfObjs, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	// TODO delete print
	printf("from slave numOfObjs = %d\n", *numOfObjs);

	*(objects) = (Obj*)malloc(sizeof(Obj)*(*numOfObjs));
	if(!*(objects))
	{
		printf("Failed to malloc objects..\n");
		exit(1);
	}
	for(int i = 0; i < (*numOfObjs); i++)
	{
		MPI_Recv(&(*objects)[i].objId, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		// TODO delete print
		printf("from slave object Id: %d\n", (*objects)[i].objId);

		MPI_Recv(&(*objects)[i].objSize, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		// TODO delete print
		printf("from slave object size: %d\n", (*objects)[i].objSize);

		(*objects)[i].objArr = (int*)malloc(sizeof(int)*(*objects)[i].objSize*(*objects)[i].objSize);
		for(int j = 0; j < (*objects)[i].objSize * (*objects)[i].objSize; j++)
		{
			MPI_Recv(&(*objects)[i].objArr[j], 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			// TODO delete print
			printf("from slave object[%d][%d] = %d\n", i, j, (*objects)[i].objArr[j]);
		}
	}

	// TODO delete print
	printf("\n\n");

	MPI_Recv(numOfPics, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	// TODO delete print
	printf("from slave numOfPics = %d\n", (*numOfPics));

	*(pictures) = (Picture*)malloc(sizeof(Picture)*(*numOfPics));
	for(int i = 0; i < (*numOfPics); i++)
	{
		MPI_Recv(&(*pictures)[i].picId, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		// TODO delete print
		printf("from slave picture id: %d\n", (*pictures)[i].picId);

		MPI_Recv(&(*pictures)[i].picSize, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		// TODO delete print
		printf("from slave picture size: %d\n", (*pictures)[i].picSize);

		(*pictures)[i].picArr = (int*)malloc(sizeof(int)*(*pictures)[i].picSize*(*pictures)[i].picSize);
		for(int j = 0; j < (*pictures)[i].picSize*(*pictures)[i].picSize; j++)
		{
			MPI_Recv(&(*pictures)[i].picArr[j], 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			printf("from slave picture[%d][%d] = %d\n", i, j, (*pictures)[i].picArr[j]);
		}
	}
}

// TODO: remove 'my_rank' parameter
void searchForMatch(Picture** pictures, Obj** objects, int* matching, int* numOfPics, int* numOfObjs, int my_rank)
{
	
	for(int i = 0; i < *numOfPics; i++)
	{
		#pragma omp parallel
		{
			#pragma omp for
			for(int j = 0; j < *numOfObjs; j++)
			{
				cudaFuncs(pictures[i], objects[j], matching);
			}
		}
	}
}