#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>
#include <omp.h>
#include "edge_detection.h"

#define NUM_THREADS 68

int decode_image(const char *srcFileName, uint32_t *input_image_width, uint32_t *input_image_height, unsigned char **input_image_data)
{
	lodepng_decode32_file(input_image_data, input_image_width, input_image_height, srcFileName);
	return 0;
}

int encode_image(const char *destFileName, uint32_t width, uint32_t height, unsigned char *output_image_data)
{
	printf("output filename: %s, width: %d, height: %d\n", destFileName, width, height);
	lodepng_encode32_file(destFileName, output_image_data, width, height);
	return 0;
}

int RGBA_to_grayscale(uint32_t input_image_width, uint32_t input_image_height, unsigned char *input_image_data, unsigned char *gray_input_image_data)
{

	uint32_t R, G, B, grayVal;
// int id = omp_get_thread_num();
#pragma omp parallel for
	for (uint32_t i = 0; i < input_image_width * input_image_height; i++)
	{
		// printf("before load\n");
		R = input_image_data[4 * i];
		G = input_image_data[4 * i + 1];
		B = input_image_data[4 * i + 2];
		grayVal = (R + G + B) / 3;
		// printf("after load, %d\n", i);
		gray_input_image_data[i] = (unsigned char)grayVal;
	}

	return 0;
}

int conv3(uint32_t image_width, uint32_t image_height, unsigned char *image_data, signed char kernel[3][3], int16_t *mat_data)
{
#pragma omp parallel for
	for (uint32_t row = 1; row < image_height - 1; row++)
	{
		for (uint32_t col = 1; col < image_width - 1; col++)
		{
			mat_data[row * image_width + col] += kernel[0][0] * image_data[(row + 1) * image_width + col + 1];
			mat_data[row * image_width + col] += kernel[0][1] * image_data[(row + 1) * image_width + col];
			mat_data[row * image_width + col] += kernel[0][2] * image_data[(row + 1) * image_width + col - 1];

			mat_data[row * image_width + col] += kernel[1][0] * image_data[row * image_width + col + 1];
			mat_data[row * image_width + col] += kernel[1][1] * image_data[row * image_width + col];
			mat_data[row * image_width + col] += kernel[1][2] * image_data[row * image_width + col - 1];

			mat_data[row * image_width + col] += kernel[2][0] * image_data[(row - 1) * image_width + col + 1];
			mat_data[row * image_width + col] += kernel[2][1] * image_data[(row - 1) * image_width + col];
			mat_data[row * image_width + col] += kernel[2][2] * image_data[(row - 1) * image_width + col - 1];
		}
	}

	uint32_t startBeforeLastRow = image_width * (image_height - 2);
	uint32_t startLastRow = image_width * (image_height - 1);
#pragma omp parallel for
	for (uint32_t col = 1; col < image_width - 1; col++)
	{
		mat_data[col] = mat_data[image_width + 1];
		mat_data[startLastRow + col] = mat_data[startBeforeLastRow + col];
	}
#pragma omp parallel for
	for (uint32_t startRow = 0; startRow < image_height; startRow += image_width)
	{
		mat_data[startRow] = mat_data[startRow + 1];
		mat_data[startRow + image_width - 1] = mat_data[startRow + image_width - 2];
	}
	return 0;
}

/* matrix --> grayscale image (0-255) */
int post_processing(uint32_t image_width, uint32_t image_height, int16_t *gradX_mat_data, int16_t *gradY_mat_data, unsigned char *gray_output_image_data)
{
	double start_post = omp_get_wtime();
	// get the max value:
	int16_t max = ~0; // min negative value: 1111111...
	int16_t *temp_mat = calloc(image_width * image_height, sizeof(int16_t));

#pragma omp parallel for
	for (uint32_t i = 0; i < image_width * image_height; i++)
	{
		temp_mat[i] = sqrt(gradX_mat_data[i] * gradX_mat_data[i] + gradY_mat_data[i] * gradY_mat_data[i]);
	}
	double start_serial = omp_get_wtime();
	for (uint32_t i = 0; i < image_width * image_height; i++)
	{
		if (temp_mat[i] > max)
		{
			max = temp_mat[i];
		}
	}
	double end_serial = omp_get_wtime();

// grayscale stretch
#pragma omp parallel for
	for (uint32_t i = 0; i < image_width * image_height; i++)
	{
		gray_output_image_data[i] = (unsigned char)((temp_mat[i] * 255) / max);
	}
	double end_post = omp_get_wtime();
	printf("Post_Processing: %f sec, including serial find max: %f sec\n", end_post - start_post, end_serial - start_serial);
	return 0;
}

int grayscale_to_RGBA(uint32_t image_width, uint32_t image_height, unsigned char *gray_output_image_data,
					  unsigned char *output_image_data)
{

#pragma omp parallel for
	for (uint32_t i = 0; i < image_width * image_height; i++)
	{
		uint32_t grayVal = gray_output_image_data[i];
		output_image_data[4 * i] = grayVal;
		output_image_data[4 * i + 1] = grayVal;
		output_image_data[4 * i + 2] = grayVal;
		output_image_data[4 * i + 3] = 255; /* fully opaque */
	}
	return 0;
}

int edge_detection(uint32_t input_image_width, uint32_t input_image_height, unsigned char *input_image_data,
				   unsigned char *output_image_data)
{
	// allocation:
	// define gray image data structure
	double start_allo = omp_get_wtime();
	unsigned char *gray_input_image_data = calloc(input_image_width * input_image_height, sizeof(unsigned char));
	unsigned char *gray_output_image_data = calloc(input_image_width * input_image_height, sizeof(unsigned char));
	// define output mat
	// before alloc
	int16_t *gradX_mat_data = calloc(input_image_width * input_image_height, sizeof(int16_t));
	int16_t *gradY_mat_data = calloc(input_image_width * input_image_height, sizeof(int16_t));
	// kernel
	signed char sobel_kernel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
	signed char sobel_kernel_y[3][3] = {{-1,-2,-1}, {0,0,0}, {1,2,1}};
	double end_allo = omp_get_wtime();
	printf("\nAlloc: %lf sec\n", end_allo - start_allo);
	// RGBA --> Gray
	double start_rgba2gray = omp_get_wtime();
	omp_set_num_threads(NUM_THREADS);
	RGBA_to_grayscale(input_image_width, input_image_height, input_image_data, gray_input_image_data);
	double end_rgba2gray = omp_get_wtime();
	printf("rgba2gray: %lf sec\n", end_rgba2gray - start_rgba2gray);

	// edge detection
	double start_conv3 = omp_get_wtime();
	conv3(input_image_width, input_image_height, gray_input_image_data, sobel_kernel_x, gradX_mat_data);
	conv3(input_image_width, input_image_height, gray_input_image_data, sobel_kernel_y, gradY_mat_data);
	double end_conv3 = omp_get_wtime();
	post_processing(input_image_width, input_image_height, gradX_mat_data, gradY_mat_data, gray_output_image_data);
	printf("conv3: %lf sec\n", end_conv3 - start_conv3);

	// Gray --> RGBA
	double start_gray2rgba = omp_get_wtime();
	grayscale_to_RGBA(input_image_width, input_image_height, gray_output_image_data, output_image_data);
	double end_gray2rgba = omp_get_wtime();
	printf("gray2rgba: %lf sec\n", end_gray2rgba - start_gray2rgba);
	return 0;
}

int main(int argc, const char *argv[])
{
	if (argc < 3) {
		printf("Please specify the input and output path\n");
		return 1;
	}

	const char *inFileName = argv[1];
	const char *outFileName = argv[2];

	// definition of data sturcture
	uint32_t width;
	uint32_t height;
	unsigned char *input_image_data;
	unsigned char *output_image_data;

	// decode
	decode_image(inFileName, &width, &height, &input_image_data);
	printf("input filename: %s, width: %d, height:%d\n", inFileName, width, height);
	output_image_data = calloc(width * height * 4, sizeof(int16_t));

	// edge detection
	// struct timeval start, stop;
	// double secs = 0;
	// gettimeofday(&start, NULL);
	double start = omp_get_wtime();
	edge_detection(width, height, input_image_data, output_image_data);
	double end = omp_get_wtime();

	// gettimeofday(&stop, NULL);
	// secs = (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);
	printf("Finished Edge Detection in %f sec\n", end - start);

	// encode
	encode_image(outFileName, width, height, output_image_data);
	printf("wrote edge image into local files\n");
	return 0;
}