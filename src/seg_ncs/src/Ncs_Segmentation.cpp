#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

extern "C" {
#include "fp16.h"
}
#include <mvnc.h>
#include "image_util.h"


// somewhat arbitrary buffer size for the device name
#define NAME_SIZE 100

// graph file name
#define GRAPH_FILE_NAME "/dl/ros/seg_ncs_ros/src/seg_ncs/bag/graph"


// 16 bits.  will use this to store half precision floats since C++ has no
// built in support for it.
typedef unsigned short half;

// image dimensions, network mean values for each channel in BGR order.
const int networkDim = 224;
float networkMean[] = {71.60167789, 82.09696889, 72.30608881};
const int target_h = 320;
const int target_w = 480;
//// movidius 设备预处理
bool g_graph_Success = false;
mvncStatus retCode;
void *deviceHandle;
char devName[NAME_SIZE];
void* graphHandle;


//namespace seg_ncs {
//
//    Ncs_Segmentation::Ncs_Segmentation(ros::NodeHandle nh)
//    : nh_(nh),
//      it_(nh_){
//
//        ROS_INFO("[SSD_Detector] Node started.");
//
//        g_graph_Success = false;
//
//        init_ncs();
//
//    }
//
//
//    Ncs_Segmentation::~Ncs_Segmentation() {
//
//        //clear and close ncs
//        ROS_INFO("Delete movidius SSD graph");
//        retCode = mvncDeallocateGraph(graphHandle);
//        graphHandle = NULL;
//
//        free(graphFileBuf);
//        retCode = mvncCloseDevice(deviceHandle);
//        deviceHandle = NULL;
//
//        ssdThread_.join();
//    }
//
//
//
//}


// Load a graph file
// caller must free the buffer returned.
void *LoadFile(const char *path, unsigned int *length)
{
    FILE *fp;
    char *buf;

    fp = fopen(path, "rb");
    if(fp == NULL)
        return 0;
    fseek(fp, 0, SEEK_END);
    *length = ftell(fp);
    rewind(fp);
    if(!(buf = (char*) malloc(*length)))
    {
        fclose(fp);
        return 0;
    }
    if(fread(buf, 1, *length, fp) != *length)
    {
        fclose(fp);
        free(buf);
        return 0;
    }
    fclose(fp);
    return buf;
}


half *LoadImage(unsigned char *img, int target_w, int target_h, int ori_w, int ori_h, float *mean)
{
    int i;
    unsigned char *imgresized;
    float *imgfp32;
    half *imgfp16;

    if(!img)
    {
        printf("The picture  could not be loaded\n");
        return 0;
    }
    imgresized = (unsigned char*) malloc(3*target_w*target_h);
    if(!imgresized)
    {
        free(img);
        perror("malloc");
        return 0;
    }
    //std::cout << "img: " << img << std::endl;
    stbir_resize_uint8(img, ori_w, ori_h, 0, imgresized, target_w, target_h, 0, 3);
    free(img);
    imgfp32 = (float*) malloc(sizeof(*imgfp32) * target_w * target_h * 3);
    if(!imgfp32)
    {
        free(imgresized);
        perror("malloc");
        return 0;
    }
    for(i = 0; i < target_w * target_h * 3; i++)
        imgfp32[i] = imgresized[i];
    free(imgresized);
    imgfp16 = (half*) malloc(sizeof(*imgfp16) * target_w * target_h * 3);
    if(!imgfp16)
    {
        free(imgfp32);
        perror("malloc");
        return 0;
    }
    //adjust values to range between -1.0 and + 1.0
    //change color channel
    for(i = 0; i < target_w*target_h; i++)
    {
        float blue, green, red;
        blue = imgfp32[3*i+2];
        green = imgfp32[3*i+1];
        red = imgfp32[3*i+0];

        imgfp32[3*i+0] = blue-mean[0];
        imgfp32[3*i+1] = green-mean[1];
        imgfp32[3*i+2] = red-mean[2];

        // uncomment to see what values are getting passed to mvncLoadTensor() before conversion to half float
        //printf("Blue: %f, Grean: %f,  Red: %f \n", imgfp32[3*i+0], imgfp32[3*i+1], imgfp32[3*i+2]);
    }

    floattofp16((unsigned char *)imgfp16, imgfp32, 3*target_w*target_h);
    free(imgfp32);
    return imgfp16;
}



//对推理结果进行展示
cv::Mat ncs_result_process(float* output, int h, int w)
{

//    out = out.reshape(-1, 2).T.reshape(2, 331, -1)
//    out = out.argmax(axis=0)
//    out = out[:-11, :-11]
//    for (int i = 20000; i < 20050 ; ++i) {
//        printf("ncs out: %f", output[i]);
//    }
    int margin = 11;
    h = h + margin;
    w = w + margin;

    cv::Mat out_img(h, w, CV_8UC1);

    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            if(output[2*(w*i + j)] < output[2*(w*i + j) + 1]){
                out_img.at<uchar>(i,j) = 255;
            } else{
                out_img.at<uchar>(i,j) = 0;
            }
        }
    }
    return out_img;

}


// callback for inference
void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        cv::Mat ROS_img = cv_bridge::toCvShare(msg, "bgr8")->image;
        cv::imshow("ori", ROS_img);

        //
//        printf("ROS_img %f\n", ROS_img)

        unsigned char *img = cvMat_to_charImg(ROS_img);
        unsigned int graphFileLen;
        half *imageBufFp16 = LoadImage(img, target_w, target_h, ROS_img.cols, ROS_img.rows, networkMean);


        // calculate the length of the buffer that contains the half precision floats.
        // 3 channels * width * height * sizeof a 16bit float
        unsigned int lenBufFp16 = 3 * target_w * target_h * sizeof(*imageBufFp16);

        // std::cout << "networkDim: " << networkDim << " imageBufFp16: " << sizeof(*imageBufFp16) << " lenBufFp16: " << lenBufFp16 << std::endl;
        std::cout << " imageBufFp16: " << *imageBufFp16 << std::endl;
        retCode = mvncLoadTensor(graphHandle, imageBufFp16, lenBufFp16, NULL);

        if (retCode != MVNC_OK) {     // error loading tensor
            perror("Could not load ssd tensor\n");
            printf("Error from mvncLoadTensor is: %d\n", retCode);
        }
        printf("Successfully loaded the tensor for image\n");

        // 判断 inference Graph的状态
        if (g_graph_Success == true) {
            void *resultData16;
            void *userParam;
            unsigned int lenResultData;
            // 执行inference
            retCode = mvncGetResult(graphHandle, &resultData16, &lenResultData, &userParam);
            if (retCode ==
                MVNC_OK) {   // Successfully got the result.  The inference result is in the buffer pointed to by resultData
                printf("Successfully got the inference result for image \n");
                printf("resultData is %d bytes which is %d 16-bit floats.\n", lenResultData,
                       lenResultData / (int) sizeof(half));

                int numResults = lenResultData / sizeof(half);
                float *resultData32;
                resultData32 = (float *) malloc(numResults * sizeof(*resultData32));
                fp16tofloat(resultData32, (unsigned char *) resultData16, numResults);

                //post process
                cv::Mat out_img = ncs_result_process(resultData32, target_h, target_w);
                cv::imshow("view", out_img);

                // 发布topic
                sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", out_img).toImageMsg();

            }
        }

    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
    cv::waitKey(10);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_listener");
    ros::NodeHandle nh;

    retCode = mvncGetDeviceName(0, devName, NAME_SIZE);
    if (retCode != MVNC_OK)
    {   // failed to get device name, maybe none plugged in.
        printf("No NCS devices found\n");
        exit(-1);
    }

    // Try to open the NCS device via the device name
    retCode = mvncOpenDevice(devName, &deviceHandle);
    if (retCode != MVNC_OK)
    {   // failed to open the device.
        printf("Could not open NCS device\n");
        exit(-1);
    }

    // deviceHandle is ready to use now.
    // Pass it to other NC API calls as needed and close it when finished.
    printf("Successfully opened NCS device!\n");

    // Now read in a graph file
    unsigned int graphFileLen;
    void* graphFileBuf = LoadFile(GRAPH_FILE_NAME, &graphFileLen);

    // allocate the graph
    retCode = mvncAllocateGraph(deviceHandle, &graphHandle, graphFileBuf, graphFileLen);
    if (retCode != MVNC_OK)
    {   // error allocating graph
        printf("Could not allocate graph for file: %s\n", GRAPH_FILE_NAME);
        printf("Error from mvncAllocateGraph is: %d\n", retCode);
        exit(-1);
    } else {
        printf("Successfully allocate graph for file: %s\n", GRAPH_FILE_NAME);
        g_graph_Success = true;
    }

//    cv::namedWindow("view");
//    cv::startWindowThread();
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("tutorial/image", 1,  imageCallback);
    ros::spin();
//    cv::destroyWindow("view");
}