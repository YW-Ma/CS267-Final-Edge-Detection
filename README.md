# Edge Detection
Edge Detection

- v2 is the updated one now. (seq only)
- reference is the reference codes (seq, omp, cuda)

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
