build:
	mpicxx -fopenmp -g -c main.c -o main.o	
	mpicxx -fopenmp -g -c functions.c -o functions.o
	mpicxx -fopenmp -o mpiCudaOpemMP  main.o functions.o
	nvcc -I./inc -I./cuda-samples-master/Common -c cudaFunctions.cu -o cudaFunctions.o
	mpicxx -fopenmp -o mpiCudaOpemMP  main.o functions.o cudaFunctions.o /usr/local/cuda-11.0/lib64/libcudart_static.a -ldl -lrt
	


clean:
	rm -f *.o ./mpiCudaOpemMP

run:
	mpiexec -np 2 ./mpiCudaOpemMP 

twoComputers:
	mpiexec -np 2 -machinefile mf -map-by node ./mpiCudaOpemMP