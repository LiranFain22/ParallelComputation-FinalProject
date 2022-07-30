#include "functions.h"
#include "mpi.h"

int main(int argc, char *argv[])
{
	int my_rank;	   /* rank of process */
	int p;			   /* number of processes */
	float matching;	   /* number of The total difference */
	int numOfPics;	   /* number of pictures */
	int numOfObjs;	   /* number of objects */
	int numOfSlavesPics;
	Picture *pictures; /* array of struct pictures */
	Obj *objects;	   /* array of struct objects */
	Match *matches;    /* array of struct matches */
	

	/* start up MPI */
	MPI_Init(&argc, &argv);

	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	printf("numOfProcess: %d\n", p);

	/* preparing data for each process */
	if (my_rank != MASTER)
	{
		runSlave(&pictures, &objects, &matching, &numOfPics, &numOfObjs, &matches);
	}
	else
	{
		runMaster(p, FILE_READ, &pictures, &objects, &matching, &numOfPics, &numOfSlavesPics, &numOfObjs, &matches);
	}

	/* search for matchs */
	searchForMatch(&pictures, &objects, &matching, &numOfPics, &numOfObjs, my_rank, &matches);

	// print master result
	// if master print 
	if(my_rank == MASTER)
	{
		for(int i = 0; i < numOfPics; i++)
		{
			printMatch(&(matches[i]));
		}
	}
	//print slave result
	// if slave send
	// if master recv and print
	printSlaveResult(matches, my_rank, numOfSlavesPics);
	

	/* free memory allocations */
	freePictures(&pictures, numOfPics);
	freeObjects(&objects, numOfObjs);

	/* shut down MPI */
	MPI_Finalize();
	return 0;
}
