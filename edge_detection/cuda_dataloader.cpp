#include "lodepng.h"
#include <string>

typedef unsigned char byte; // most useful typedef ever

struct imgData {
    imgData(byte* pix = nullptr, unsigned int w = 0, unsigned int h = 0) : pixels(pix), width(w), height(h) {
    };
    byte* pixels;
    unsigned int width;
    unsigned int height;
};