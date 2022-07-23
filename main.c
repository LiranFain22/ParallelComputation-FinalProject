#include "functions.h"
#include "mpi.h"

int main(int argc, char *argv[])
{
	int my_rank;	   /* rank of process */
	int p;			   /* number of processes */
	float matching;	   /* number of The total difference */
	int numOfPics;	   /* number of pictures */
	int numOfObjs;	   /* number of objects */
	Picture *pictures; /* array of struct pictures */
	Obj *objects;	   /* array of struct objects */

	/* start up MPI */
	MPI_Init(&argc, &argv);

	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	/* preparing data from each process */
	if (my_rank != 0)
	{
		runSlave(&pictures, &objects, &matching, &numOfPics, &numOfObjs);
		// printf("slave got numOfPic = %d, picture[3] = %d, picture[4] = %d\n",numOfPics, pictures[0].picArr[3], pictures[0].picArr[4]);
	}
	else
	{
		runMaster(p, "/home/linuxu/ParallelComputationFinalProject/input.txt", &pictures, &objects, &matching, &numOfPics, &numOfObjs);
		// printf("master got numOfPic = %d, picture[3] = %d, picture[4] = %d\n",numOfPics, pictures[0].picArr[3], pictures[0].picArr[4]);
	}
	// TODO - delete print
	// printf("start searchForMatch function:\n\n");
	/* --- search for match --- */
	searchForMatch(&pictures, &objects, &matching, &numOfPics, &numOfObjs);

	/* shut down MPI */
	MPI_Finalize();
	return 0;
}
