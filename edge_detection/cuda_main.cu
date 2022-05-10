#include <thread>
#include <chrono>
#include <time.h>
#include <iostream>
#include <math.h>
#include <string>
#include "lodepng.h"

typedef unsigned char byte;
struct pixel {
    unsigned char R;
    unsigned char G;
    unsigned char B;
    unsigned char A;
};

static inline uint32_t getNextPowerOf2(uint32_t n) {
    uint32_t cur = n;
    uint32_t rslt = 1;
    /* Find the previous power of 2 */
    while (cur >>= 1) {
        rslt <<= 1;
    }

    if (rslt == n) {
        return n;
    } else {
        return (rslt << 1);
    }
}

__global__ void edge_detection(const byte* orig, int* cpu, const unsigned int width, const unsigned int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float dx, dy;
    if( x > 0 && y > 0 && x < width-1 && y < height-1) {
        dx = (-1* orig[(y-1)*width + (x-1)]) + (-2*orig[y*width+(x-1)]) + (-1*orig[(y+1)*width+(x-1)]) +
             (    orig[(y-1)*width + (x+1)]) + ( 2*orig[y*width+(x+1)]) + (   orig[(y+1)*width+(x+1)]);
        dy = (-1* orig[(y-1)*width + (x-1)]) + (-2*orig[(y-1)*width+x]) + (-1*orig[(y-1)*width+(x+1)]) +
             (    orig[(y+1)*width + (x-1)]) + ( 2*orig[(y+1)*width+x]) + (   orig[(y+1)*width+(x+1)]);
        cpu[y*width + x] = sqrt( (dx*dx) + (dy*dy) );
  		// Modified dy
    }
}

__global__ void rgba_to_grayscale(pixel *rgba, byte* gray, const unsigned int width, const unsigned int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	if( x >= 0 && y >= 0 && x < width && y < height) {
		int sum = 0;
		int id = y*width+x;
		sum += rgba[id].R;
		sum += rgba[id].G;
		sum += rgba[id].B;
		gray[id] = sum / 3;
    }
}

__global__ void max_reduction_kernel(int *pMaxGrads, const unsigned int width,
                                     const unsigned int height, uint32_t nPxPerThread)
{
    extern __shared__ uint16_t sData[]; /* Contains the data for the local reduction */

    uint32_t pxNum = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid = threadIdx.x;

    /* Each thread copies its pixel, and maybe more if needed.
       If there is no pixel, then the value 0 is chosen (as it is neutral for a max) */
    if (pxNum < width * height) {
        sData[tid] = pMaxGrads[pxNum];
    } else {
        sData[tid] = 0;
    }

    for (int i = 1; i < nPxPerThread; i++) {
        uint32_t nextPxIdx = pxNum + blockDim.x * gridDim.x * i;
        if (nextPxIdx < width * height) {
            sData[tid] = max(sData[tid], pMaxGrads[nextPxIdx]);
        }
    }

    __syncthreads();

    /* Now, reduce in parallel to find the max */
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride = stride >> 1) {
        if (tid < stride) { /* If we are on the lowest part of the remaining array */
            sData[tid] = max(sData[tid], sData[tid + stride]);
        }

        __syncthreads();
    }


    if (tid == 0) {
        pMaxGrads[blockIdx.x] = sData[0];
    }

    /* At the end of that kernel, pMaxGrads[blockIdx.x] contains the max value of each block.
       We can recursively call the kernel on the smaller resulting array. */
}

__global__ void normalize_gpu(int* max, int* edge, struct pixel* output, const unsigned int width, const unsigned int height, uint32_t basePx) {
    
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t pxNum = y * width + x + basePx;
    int maxGrad = max[0];

    int grayVal = int(float(edge[pxNum]) / maxGrad * 255);
    output[pxNum].R = grayVal;
    output[pxNum].G = grayVal;
    output[pxNum].B = grayVal;
    output[pxNum].A = 255;
}

int main(int argc, char*argv[]) {
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	// 1. Decoding (CPU)
	unsigned int width, height;
    byte* input;
    unsigned error = lodepng_decode_file(&input, &width, &height, argv[1], LCT_RGBA, 8);
	pixel* cpu_rgba = (pixel*)calloc(width * height, sizeof(pixel));
	for (int i = 0; i < width * height; i++) {
		cpu_rgba[i].R = input[4*i];
		cpu_rgba[i].G = input[4*i + 1];
		cpu_rgba[i].B = input[4*i + 2];
		cpu_rgba[i].A = input[4*i + 3];
	}
	// 2. GPU Memory preparation
    pixel *gpu_rgba;
	cudaMalloc((void **)&gpu_rgba, (width * height) * sizeof(pixel));
	cudaMemcpy(gpu_rgba, cpu_rgba, (width * height) * sizeof(pixel), cudaMemcpyHostToDevice);
	byte *gpu_orig;
    int *gpu_sobel;
    cudaMalloc((void **)&gpu_orig, (width * height) * sizeof(byte));
    cudaMalloc((void **)&gpu_sobel, (width * height) * sizeof(int));
    cudaMemset(gpu_sobel, 0, (width * height) * sizeof(int));
   
    // 3. Setup CUDA property, Grid = 20
    dim3 threadsPerBlock(20, 20, 1);
    dim3 numBlocks(ceil(width/20), ceil(height/20), 1);
	printf("threadsPerBlock = 400\n");

    auto start_time = std::chrono::system_clock::now();
	// 4. RGBA --> Grayscale
	rgba_to_grayscale<<<numBlocks, threadsPerBlock>>>(gpu_rgba, gpu_orig, width, height);
	std::chrono::duration<double> rgba2gray = std::chrono::system_clock::now() - start_time;
	printf("Finish RGBA->Grayscale in: %fms\n", 1000 * rgba2gray.count());
	// 5. sobel edge detection.
    edge_detection<<<numBlocks, threadsPerBlock>>>(gpu_orig, gpu_sobel, width, height);
    cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    if ( cudaerror != cudaSuccess ) fprintf( stderr, "Cuda failed to synchronize: %s\n", cudaGetErrorName( cudaerror ) ); // if error, output error
    
	// 6. Grayscale Stretching
	// 6.1 get max
    int *maxGradsDevice = NULL;
    cudaMalloc((void **)&maxGradsDevice, width * height * sizeof(int));
    cudaMemcpy(maxGradsDevice, gpu_sobel, width * height * sizeof(int), cudaMemcpyDeviceToDevice);
    int maxThreadsPerBlock = devProp.maxThreadsDim[0];
    int maxConcurrentBlocks = devProp.maxGridSize[0];
    int maxConcurrentThreads = maxConcurrentBlocks * maxThreadsPerBlock;
    unsigned int remainingElems = width * height;
    while (remainingElems > 1) {
        uint32_t threadsPerBlock = min(remainingElems, maxThreadsPerBlock);
        threadsPerBlock = getNextPowerOf2(threadsPerBlock);
        /* Don't allocate more blocks than possible. If there are too many pixels,
            some blocks will handle several pixels (handled in the kernel). */
        uint32_t nBlocks = min(remainingElems / threadsPerBlock + (remainingElems % threadsPerBlock == 0 ? 0 : 1), maxConcurrentBlocks);
        uint32_t nThreads = threadsPerBlock * nBlocks;
        uint32_t nPxPerThread = remainingElems / nThreads + (height * width % nThreads == 0 ? 0 : 1);
        uint32_t sharedMem = threadsPerBlock * sizeof(uint16_t);

        max_reduction_kernel <<< nBlocks, threadsPerBlock, sharedMem >>>(maxGradsDevice, width, height, nPxPerThread);
        /* One remaining element by running block */
        remainingElems = nBlocks;
    }
	// 6.2 Prepare normalized mat
    uint16_t *outNormalized = NULL;
    cudaMalloc((void **) &outNormalized, width * height * sizeof(struct pixel));
    struct pixel *outNormalizedDevice = NULL;
    cudaMalloc((void **) &outNormalizedDevice, width * height * sizeof(struct pixel));
    for (uint32_t basePx = 0; basePx < height * width; basePx += maxConcurrentThreads) {

        /* Don't use more blocks than necessary */
        uint32_t runningThreads = min(height * width - basePx, maxConcurrentThreads);
        uint32_t nBlocks = runningThreads / maxThreadsPerBlock +
                            (runningThreads % maxThreadsPerBlock == 0 ? 0 : 1);
    
	// 7. GrayScale to RGBA also in normalization part.
        normalize_gpu <<< nBlocks, maxThreadsPerBlock >>>
            (maxGradsDevice, gpu_sobel, outNormalizedDevice, width, height, basePx);
    }
    std::chrono::duration<double> edge_detection_all = std::chrono::system_clock::now() - start_time;

	// 8. Copy back to CPU and encode to .PNG file.
    struct pixel *output_image_data = (struct pixel*)calloc(width * height, sizeof(struct pixel));
    cudaMemcpy(output_image_data, outNormalizedDevice, width * height * sizeof(struct pixel), cudaMemcpyDeviceToHost);
    printf("\n Input file %s, width: %d, height: %d \n", argv[1], width, height);
    printf("Finish Edge Detection in = %f msec\n", 1000 * edge_detection_all.count());
    unsigned char *output_image = (unsigned char*)calloc(width * height * 4, sizeof(unsigned char));
    for(int i = 0; i < width * height; i++){
        output_image[i * 4] = output_image_data[i].R;
        output_image[i * 4 + 1] = output_image_data[i].G;
        output_image[i * 4 + 2] = output_image_data[i].B;
        output_image[i * 4 + 3] = output_image_data[i].A;
    }
    lodepng_encode32_file(argv[2], output_image, width, height);
    cudaFree(gpu_orig); cudaFree(gpu_sobel);
    return 0;
}
