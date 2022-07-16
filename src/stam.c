// #include "functions.h"
// #include "sequenceAlignment.h"

// int main(int argc, char *argv[])
// {
//     //Variables
//     FILE *fp;
//     char *seq1 = NULL;
//     char **seq2Arr = NULL;
//     float *weightArr = NULL;
//     int numberOfSeq2;
//     int my_rank, p;
//     Score *bestMutantArr;

//     //Initialize MPI settings
//     MPI_settings(&argc, &argv, &my_rank, &p);
//     if (p != 2)
//     {
//         printf("Number of process must be 2!\n");fflush(stdout);
//         MPI_Abort(MPI_COMM_WORLD, 0);
//     }

//     if (my_rank == MASTER)
//     {
//         //Reads data from the file
//         readFromFile(&seq1, &weightArr, &seq2Arr, &numberOfSeq2, fp);

//         MPI_Send(&numberOfSeq2, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
//         MPI_Send(weightArr, WEIGHTS, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
//         MPI_Send(seq1, N1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

//         for (int i = 0; i < numberOfSeq2; i++)
//         {
//             MPI_Send(seq2Arr[i], N2, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
//         }
//     }
//     else
//     {
//         //Variables
//         char seq1Recv[N1];
//         char seq2Recv[N2];

//         weightArr = (float*)malloc(sizeof(float)*WEIGHTS);
//         if (!weightArr)
//             exit(1);
        
//         MPI_Recv(&numberOfSeq2, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         MPI_Recv(weightArr, WEIGHTS, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         MPI_Recv(seq1Recv, N1, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

//         seq2Arr = (char**)malloc(sizeof(char*)*(numberOfSeq2));
//         seq1 = strdup(seq1Recv);

//         if (!seq1 || !seq2Arr)
//             exit(1);

//         for (int i = 0; i < numberOfSeq2; i++)
//         {
//             MPI_Recv(seq2Recv, N2, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//             seq2Arr[i] = strdup(seq2Recv);
//             if (!seq2Arr[i])
//                 exit(1);
//         }                      
//     }
    
//     bestMutantArr = (Score*)calloc(4 ,sizeof(Score));
//     if (!bestMutantArr)
//         exit(1);

//     //--------------------Starting The Test--------------------
//     greatestMutant(seq1, seq2Arr, weightArr, bestMutantArr, my_rank);

//     if (my_rank == MASTER)
//     {
//         for (int i = 0; i < numberOfSeq2; i++)
//         {
//             printf("Offset = %d, (%d, %d), Score %f\n", bestMutantArr[i].offset, bestMutantArr[i].n, bestMutantArr[i].k, bestMutantArr[i].alignmentScore);
//         }
//     }
    
//     //Free memory
//     free(bestMutantArr);
//     free(seq1);
//     for (int i = 0; i < numberOfSeq2; i++)
//     {
//         free(seq2Arr[i]);
//     }
//     free(seq2Arr);
//     free(weightArr);

//     MPI_Finalize();
//     return 0;
// }