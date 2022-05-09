# Edge Detection
Author:
    Guanyue Bian,
    Jingjing Wei,
    Yaowei Ma

Reference:
    Image Read / Write library:
    Image Decode / Encode:
    Algorithm to Get Maximum Number in CUDA:
    


# Please put .png images under images folder
# Build:
    make clean
    +
    make openmp
    make serial
    make cuda

# Run:
    ./serial {input image path} {output image path}
    
    
    #OpenMP:
    #OpenMP settings:
    export OMP_NUM_THREADS=68
    export OMP_PLACES=cores
    export OMP_PROC_BIND=spread

    #run the application:
    srun -n 1 ./omp_version ./../images/library.png lib.png
    srun -n 1 ./openmp ./../images/library.png lib.png

# Clean
    make clean
