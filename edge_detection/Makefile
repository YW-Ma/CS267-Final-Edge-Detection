DEPENDANCY = ./lodepng/lodepng.o
INCLUDES = -I. -I./lodepng/
CFLAGS = -O3 -std=gnu99 -Wall -Wextra -c -fopenmp $(INCLUDES)
FLAGS = -Wno-deprecated-gpu-targets -O2 -Xcompiler -fopenmp -std=c++11

openmp: omp.o $(DEPENDANCY) edge_detection.h
	gcc -lgomp -lm omp.o $(DEPENDANCY) -o $@

serial: serial.o $(DEPENDANCY) edge_detection.h
	gcc -lgomp -lm serial.o $(DEPENDANCY) -o $@

cuda: cuda_main.cu lodepng.cpp 
	nvcc -o cuda $(FLAGS) $^

clean:
	rm -f cuda
	rm -f serial
	rm -f openmp
	rm -f *.o *~ core
	rm -f output.png
	rm -f ./lodepng/lodepng.o
