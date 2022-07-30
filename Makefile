build:	
	mpicxx -fopenmp -g -c main.c -o main.o	
	nvcc -I./inc -c cudaFunctions.cu -o cudaFunctions.o
	mpicxx -fopenmp -g -c functions.c -o functions.o
	mpicxx -fopenmp -o mpiCudaOpemMP  main.o functions.o cudaFunctions.o /usr/local/cuda-11.0/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o ./mpiCudaOpemMP

run:
	mpiexec -np 2 ./mpiCudaOpemMP > result.txt

twoComputers:
	mpiexec -np 2 -machinefile machines.txt -map-by node ./mpiCudaOpemMP