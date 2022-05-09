#ifndef __EDGE_DETECTION_H__
#define __EDGE_DETECTION_H__

#include <stdio.h>
#include <stdint.h>

int decode_image(const char *srcFileName, uint32_t *input_image_width, uint32_t *input_image_height, unsigned char **input_image_data);
int encode_image(const char *destFileName, uint32_t output_image_width, uint32_t output_image_height, unsigned char *output_image_data);
int edge_detection(uint32_t input_image_width, uint32_t input_image_height, unsigned char *input_image_data, unsigned char *output_image_data);
#endif
