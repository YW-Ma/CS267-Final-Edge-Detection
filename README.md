# Edge Detection
Author:
- Guanyue Bian,
- Jingjing Wei,
- Yaowei Ma

# Instructions:
## serial and openmp
### 1. Configuration


### 2. Build
for serial:
```
make clean
make serial
```
for openmp
```
1. set the NUM_THREADS at the line 8 of omp.c to the value you want (from 1 to 68 in this experiment)
2. build
make clean
make openmp
```
### 3. Run
./{exe} {input image path} {output image path}


for serial:
```
./serial ../images/Gate_16.png output_16.png
```
for openmp
```
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
srun -n 1 ./openmp ../images/Gate_16.png output_16.png
```

## CUDA
### 1. Configuration

### 2. Build
for CUDA
```
make clean
make cuda
```
### 3. Run
```
```
# Results:


# Reference:
- Image Decode / Encode library: https://github.com/lvandeve/lodepng
- CUDA max reduction: https://github.com/Ricordel/parallel-sobel
