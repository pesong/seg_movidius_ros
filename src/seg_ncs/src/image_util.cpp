//
// Created by pesong on 18-9-5.
//

#include "image_util.h"


image ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image out = make_image(w, h, c);
    ipl_into_image(src, out);
    return out;
}


void ipl_into_image(IplImage* src, image im)
{
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    int i, j, k;
    // std::cout << "h: " << h << " w: " << w << " c: " << c << std::endl;
    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){

                //std::cout << "index: "<< k*w*h + i*w + j << " data: " << data[i*step + j*c + k]/255. << " i: " << i << " k："<< k << " j:" << j << " w: "<< w <<" h: "<<h <<" c: "<< c << std::endl;
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
}

image make_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
//  out.data = calloc(h*w*c, sizeof(float));
    return out;
}

image make_empty_image(int w, int h, int c)
{
    image out;
//  out.data = 0;
    out.data = new float[h*w*c]();//calloc(h*w*c, sizeof(float));
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

void rgbgr_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        float swap = im.data[i];
        im.data[i] = im.data[i+im.w*im.h*2];
        im.data[i+im.w*im.h*2] = swap;
    }
}

unsigned char* image_to_stb(image in)
{
    int i,j,k;
    int w = in.w;
    int h = in.h;
    int c =3;
    unsigned char *img = (unsigned char*) malloc(c*w*h);
    for(k = 0; k < c; ++k){
        for(j=0; j <h; ++j){
            for(i=0; i<w; ++i){
                int src_index = i + w*j + w*h*k;
                int dst_index = k + c*i + c*w*j;
                img[dst_index] = (unsigned char)(255*in.data[src_index]);
            }
        }
    }
    // std::cout << "xxxxx" << std::endl;
    return img;
}



unsigned char* cvMat_to_charImg(cv::Mat pic)
{
    IplImage copy = pic;
    IplImage *pic_Ipl = &copy;
    //std::cout << "pic_Ipl.width: " << pic_Ipl->width << " pic_Ipl.height: " << pic_Ipl->height << std::endl;
    //cvSaveImage("/home/ziwei/human_track_ssd/aaa.jpg",pic_Ipl);
//    ipl_into_image(pic_Ipl, buff_);
    unsigned char* pic_final = image_to_stb(ipl_to_image(pic_Ipl));

    return pic_final;
}