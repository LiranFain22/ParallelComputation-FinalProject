#include "functions.h"
#include "mpi.h"
#include <netdb.h>
#include <ifaddrs.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

void parseFile(char* path, float* matchingValue, int* numOfPics, Picture** pictures, int* numOfObjs, Obj** objects)
{
	FILE* fp;

	if((fp = fopen(path, "r")) == 0)
	{
		perror("Could not open file\n");
		exit(1);
	}

	if(fscanf(fp, "%f %d", matchingValue, numOfPics) == EOF)
	{
		perror("Could not read matching and numOfPics from file!\n");fflush(stdout);
		exit(1);
	}

	// Creating size of pictures
	*(pictures) = (Picture*)malloc(sizeof(Picture)*(*numOfPics));
	if(!*(pictures))
	{
		perror("Failed to create pictures..\n");
		exit(1);
	}

	// Inserting numbers to each picture
	for(int i = 0; i < *(numOfPics); i++)
	{
		fscanf(fp, "%d", &(*pictures)[i].picId);
		// Reads from file each picture size
		fscanf(fp, "%d", &(*pictures)[i].picSize);

		(*pictures)[i].picArr = (int*)malloc(sizeof(int)*(*pictures)[i].picSize*(*pictures)[i].picSize);
		for(int j = 0; j < (*pictures)[i].picSize * (*pictures)[i].picSize; j++)
		{
			fscanf(fp, "%d", &(*pictures)[i].picArr[j]);
		}
	}

	if(fscanf(fp, "%d", numOfObjs) == EOF)
	{
		perror("Could not read numOfObjs from file!\n");fflush(stdout);
		exit(1);
	}

	// Creating size of objects
	*(objects) = (Obj*)malloc(sizeof(Obj)*(*numOfObjs));
	if(!*(objects))
	{
		perror("Failed to create objects..\n");
		exit(1);
	}

	// Inserting numbers to each object
	for(int  i = 0; i < *(numOfObjs); i++)
	{
		fscanf(fp, "%d", &(*objects)[i].objId);

		// Read from file each object size
		fscanf(fp, "%d", &(*objects)[i].objSize);

		(*objects)[i].objArr = (int*)malloc(sizeof(int)*(*objects)[i].objSize*(*objects)[i].objSize);
		for(int j = 0; j < (*objects)[i].objSize * (*objects)[i].objSize; j++)
		{
			fscanf(fp, "%d", &(*objects)[i].objArr[j]);
		}
	}
}

void freePictures(Picture** pictures, int numOfPics)
{
    for(int i = 0; i < numOfPics; i++)
    {
		free((*pictures)[i].picArr);
    }
	free(*pictures);
}

void freeObjects(Obj** objects, int numOfObjs)
{
    for(int i = 0; i < numOfObjs; i++)
    {
		free((*objects)[i].objArr);
    }
	free(*objects);
}

void runMaster(int p, char* path, Picture** pictures, Obj** objects, float* matching, int* numOfPics, int* numOfSlavePics, int* numOfObjs, Match** matches)
{
	Picture* allPictures;
	int numOfAllPictures;

	parseFile(path, matching, &numOfAllPictures, &allPictures, numOfObjs, objects);

	if (p > numOfAllPictures)
	{
		p = numOfAllPictures;
	}

	int portionSize = numOfAllPictures / p;
	int masterPortionSize = (numOfAllPictures - (portionSize*p)) + portionSize;
	*numOfPics = masterPortionSize;
	*numOfSlavePics = portionSize;

	for(int i = 0; i < p; i++)
	{
		if (i == MASTER)
		{
			*pictures = (Picture*)malloc(sizeof(Picture)*(*numOfPics));
			if(!*pictures)
			{
				perror("Failed to create pictures..\n");
				exit(1);
			}

			*matches = (Match*)malloc(sizeof(Match)*(*numOfPics));

			for(int j = 0; j < (*numOfPics); j++)
			{
				(*pictures)[j].picId = allPictures[j].picId;
				(*pictures)[j].picSize = allPictures[j].picSize;
				(*pictures)[j].picArr = (int*)malloc(sizeof(int)*(*pictures)[j].picSize*(*pictures)[j].picSize);
				for(int k = 0; k < (*pictures)[j].picSize * (*pictures)[j].picSize; k++)
				{
					(*pictures)[j].picArr[k] = allPictures[j].picArr[k];
				}
			}
		} 
		else
		{
			MPI_Send(matching, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
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
			for(int j = masterPortionSize + (portionSize * (i-1)); j < masterPortionSize + (portionSize * i) ; j++)
			{
				MPI_Send(&(allPictures)[j].picId, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

				MPI_Send(&(allPictures)[j].picSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				for(int k = 0; k < (allPictures)[j].picSize * (allPictures)[j].picSize; k++)
				{
					MPI_Send(&(allPictures)[j].picArr[k], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				}
			}
		}
	}
	freePictures(&allPictures, numOfAllPictures);
}

void runSlave(Picture** pictures, Obj** objects, float* matching, int* numOfPics, int* numOfObjs, Match** matches)
{

	MPI_Recv(matching, 1, MPI_FLOAT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	MPI_Recv(numOfObjs, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	*(objects) = (Obj*)malloc(sizeof(Obj)*(*numOfObjs));
	if(!*(objects))
	{
		perror("Failed to malloc objects..\n");
		exit(1);
	}

	
	for(int i = 0; i < (*numOfObjs); i++)
	{
		MPI_Recv(&(*objects)[i].objId, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Recv(&(*objects)[i].objSize, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		(*objects)[i].objArr = (int*)malloc(sizeof(int)*(*objects)[i].objSize*(*objects)[i].objSize);
		for(int j = 0; j < (*objects)[i].objSize * (*objects)[i].objSize; j++)
		{
			MPI_Recv(&(*objects)[i].objArr[j], 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}


	MPI_Recv(numOfPics, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	*(pictures) = (Picture*)malloc(sizeof(Picture)*(*numOfPics));

	*matches = (Match*)malloc(sizeof(Match)*(*numOfPics));

	for(int i = 0; i < (*numOfPics); i++)
	{
		MPI_Recv(&(*pictures)[i].picId, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Recv(&(*pictures)[i].picSize, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		(*pictures)[i].picArr = (int*)malloc(sizeof(int)*(*pictures)[i].picSize*(*pictures)[i].picSize);
		for(int j = 0; j < (*pictures)[i].picSize*(*pictures)[i].picSize; j++)
		{
			MPI_Recv(&(*pictures)[i].picArr[j], 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}
}


void printMatch(Match* myMatch)
{
	if(myMatch->isMatch == 1)
		printf("Picture %d found Object %d in Position(%d,%d)\n", myMatch->picId, myMatch->objectId, myMatch->row, myMatch->col);
	else
	{
		printf("Picture %d No Objects were found\n", myMatch->picId);
	}
}

void createMatchType(MPI_Datatype *matchType)
{
    int block_length[5] = {1, 1, 1, 1, 1};
    MPI_Aint disp[5];
    MPI_Datatype types[5] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT};

    disp[0] = offsetof(Match, isMatch);
    disp[1] = offsetof(Match, objectId);
    disp[2] = offsetof(Match, picId);
    disp[3]  = offsetof(Match, row);
    disp[4]  = offsetof(Match, col);

    MPI_Type_create_struct(5, block_length, disp, types, matchType);
    MPI_Type_commit(matchType);
}

void printSlaveResult(Match* matches, int my_rank, int numOfSlavesPics)
{
	Match match;
	MPI_Datatype MatchType;
    createMatchType(&MatchType);

	if(my_rank == MASTER)
	{
		for(int i = 0; i < numOfSlavesPics; i++)
		{
			MPI_Recv(&match, 1, MatchType, SLAVE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			printMatch(&match);
		}
	}
	else
	{
		for(int i = 0; i < numOfSlavesPics; i++)
		{
			MPI_Send(&(matches[i]), 1, MatchType, MASTER, 0, MPI_COMM_WORLD);
		}
	}
}


void searchForMatch(Picture** pictures, Obj** objects, float* matching, int* numOfPics, int* numOfObjs, int my_rank, Match** matches)
{
	int foundMatch = 0;
	Match myMatch;
	Match finalMatch;
	finalMatch.isMatch = 0;


	for(int i = 0; i < *numOfPics; i++)
	{
		foundMatch = 0;
		(*matches)[i].isMatch = 0;
		(*matches)[i].picId = (*pictures)[i].picId;
		#pragma omp parallel
		{
			#pragma omp for private(myMatch)
			for(int j = 0; j < *numOfObjs; j++)
			{
				myMatch.isMatch = 0;
				if(foundMatch == 0)
				{
					cudaFuncs(&(*pictures)[i], &(*objects)[j], matching, &myMatch);

				}
				else
					continue;

				#pragma omp critical
					if (foundMatch == 0 && myMatch.isMatch == 1)
					{
						foundMatch = 1;
						(*matches)[i] = myMatch;
					}
			}
			
		}
	}

}