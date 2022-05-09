# Edge Detection
Author:
- Guanyue Bian (guanyue_bian@berkeley.edu)
- Jingjing Wei (jingjingwei@berkeley.edu)
- Yaowei Ma    (yaowei_ma@berkeley.edu)

## Overview:
Implemented the Sobel Edge Detection in serial, openmp and CUDA.\
We use the photo of the Sather Gate as an example.\
![](./readme_figures/figure_1.png)
For scaling experiments, we need to increase the pixel number of input data. So we repeat the photo and get a photo with more pixels.\
![](./readme_figures/figure_2.png)

The Experiment results was shown in the report: https://docs.google.com/document/d/1t7FT0fo983CncVQVrAsNiGYo18_BmD2c5cgNx40--iA/edit?usp=sharing (Use your UC Berkeley account to open it)

# Instructions:
## [1] serial and openmp
### 1.1 Configuration
Using your account connect to Bridges2:
```
ssh -l {your_name} -i ~/.ssh/nersc cori.nersc.gov
module swap PrgEnv-intel PrgEnv-gnu
module load cmake
```
Then salloc a GPU node
```
salloc -N 1 -C knl -q interactive -t 01:00:00
```
Then clone the code and enter the folder holding the makefile
```
git clone https://github.com/YW-Ma/CS267-Final-Edge-Detection.git
cd CS267-Final-Edge-Detection
cd edge_detection
```

### 1.2 Build
for serial:
```
make clean
make serial
```
for openmp
```
1. set the NUM_THREADS at the line 8 of omp.c to the value you want (from 4 to 68 in this experiment)
2. build
make clean
make openmp
```
### 1.3 Run
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

## [2] CUDA
### 1.1 Configuration
Using your account connect to Bridges2:
```
ssh -p 2222 {your_name}@bridges2.psc.xsede.org
```
Then salloc a GPU node
```
salloc -N 1 -p GPU-shared --gres=gpu:1 -q interactive -t 02:00:00
```
Then clone the code and enter the folder holding the makefile
```
git clone https://github.com/YW-Ma/CS267-Final-Edge-Detection.git
cd CS267-Final-Edge-Detection
cd edge_detection
module load cuda
module load cmake
```

### 1.2 Build
for CUDA
```
make clean
make cuda
```
### 1.3 Run
```
./cuda ../images/Gate_16.png output_16.png
```


# Reference:
- Image Decode / Encode library: https://github.com/lvandeve/lodepng
- CUDA max reduction: https://github.com/Ricordel/parallel-sobel
