Name: Liran Fainshtein
ID: 313583528

Description:
We will solve the problem with two processes.
Each process will find his matches between objects and pictures by using multi thread (OMP).
After calcultions Master process will print both processes results.


Parallelism idea:
MPI - two processes will work on half amount of pictures.
OpenMP - each process will handle 4 threads that will run as the amount of objects.
CUDA - each thread from OpenMP will calculate if there is a match, CUDA will
handle the amount of threads as the amount of members of each picture and each thread will calculates his result,
then it will find match by atomic functions.