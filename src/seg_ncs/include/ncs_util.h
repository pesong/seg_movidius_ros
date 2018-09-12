//
// Created by pesong on 18-9-5.
// used to process the image format
//

#ifndef NCS_UTIL_H
#define NCS_UTIL_H
#define NAME_SIZE 100

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <cv_bridge/cv_bridge.h>

#include <mvnc.h>
extern "C" {
#include "fp16.h"
}

extern  bool g_graph_Success;
extern  mvncStatus retCode;
extern  void *deviceHandle;
extern  char devName[NAME_SIZE];
extern  void* graphHandle;
extern  void* graphFileBuf;

// 16 bits.  will use this to store half precision floats since C++ has no
// built in support for it.
typedef unsigned short half;

class image {
public:
    int w;
    int h;
    int c;
    float *data;

    image() {
        data = nullptr;
    };

    image(int _w, int _h, int _c) {
        w = _w;
        h = _h;
        c = _c;
        data = new float[h * w * c]();
    };

    ~image() {
        delete[] data;
    };
};


extern image *ipl_to_image(IplImage *src);
extern void ipl_into_image(IplImage *src, image *im);

extern unsigned char* image_to_stb(image* in);
extern unsigned char* cvMat_to_charImg(cv::Mat pic);
extern void *LoadFile(const char *path, unsigned int *length);
extern half *LoadImage(unsigned char *img, int target_w, int target_h, int ori_w, int ori_h, float *mean);



#endif //NCS_UTIL_H
