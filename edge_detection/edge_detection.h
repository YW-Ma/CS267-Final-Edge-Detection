#include <stdio.h>
#include <stdint.h>

int decode_image(const char *srcFileName, uint32_t *input_image_width, uint32_t *input_image_height, unsigned char **input_image_data);
int encode_image(const char *destFileName, uint32_t output_image_width, uint32_t output_image_height, unsigned char *output_image_data);
int edge_detection(uint32_t input_image_width, uint32_t input_image_height, unsigned char *input_image_data, unsigned char *output_image_data);
int conv3(uint32_t image_width, uint32_t image_height, unsigned char *image_data, signed char kernel[3][3], int16_t *mat_data);
int RGBA_to_greyScale(uint32_t input_image_width, uint32_t input_image_height, unsigned char *input_image_data, unsigned char *grey_input_image_data);
int greyScale_to_RGBA(uint32_t image_width, uint32_t image_height, unsigned char *grey_output_image_data,unsigned char *output_image_data);
int post_processing(uint32_t image_width, uint32_t image_height, int16_t *gradX_mat_data, int16_t *gradY_mat_data, unsigned char *grey_output_image_data);
/*Same as lodepng_decode_file, but always decodes to 32-bit RGBA raw image.*/
unsigned lodepng_decode32_file(unsigned char** out, unsigned* w, unsigned* h,
                               const char* filename);
/*Same as lodepng_encode_file, but always encodes from 32-bit RGBA raw image.*/
unsigned lodepng_encode32_file(const char* filename,
                               const unsigned char* image, unsigned w, unsigned h);