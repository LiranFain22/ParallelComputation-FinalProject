#include "functions.h"
#include "mpi.h"

int main(int argc, char *argv[])
{
	int my_rank;	   /* rank of process */
	int p;			   /* number of processes */
	float matching;	   /* number of The total difference */
	int numOfPics;	   /* number of pictures */
	int numOfObjs;	   /* number of objects */
	int numOfSlavesPics; /* number of pictures in each slave */
	Picture *pictures; /* array of struct pictures */
	Obj *objects;	   /* array of struct objects */
	Match *matches;    /* array of struct matches */
	

	/* start up MPI */
	MPI_Init(&argc, &argv);

	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	/* preparing data for each process */
	if (my_rank != MASTER)
	{
		runSlave(&pictures, &objects, &matching, &numOfPics, &numOfObjs, &matches);
	}
	else
	{
		runMaster(p, FILE_READ, &pictures, &objects, &matching, &numOfPics, &numOfSlavesPics, &numOfObjs, &matches);
	}

	/* search for matches */
	searchForMatch(&pictures, &objects, &matching, &numOfPics, &numOfObjs, my_rank, &matches);

	/* 
	*	1) master process will print matches (if there is some)
	*/
	if(my_rank == MASTER)
	{
		for(int i = 0; i < numOfPics; i++)
		{
			printMatch(&(matches[i]));
		}
	}

	/* 
	*	2) slave process will send matches to master process
	*   3) master process will print the rest matches.
	*/
	if(my_rank == MASTER)
	{
		printSlaveResult(matches, my_rank, numOfSlavesPics);
	}
	else
	{
		printSlaveResult(matches, my_rank, numOfPics);
	}
	

	/* free memory allocations */
	freePictures(&pictures, numOfPics);
	freeObjects(&objects, numOfObjs);

	/* shut down MPI */
	MPI_Finalize();
	return 0;
}
