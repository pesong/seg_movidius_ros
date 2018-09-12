#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <stdio.h>
#include <stdlib.h>
#include <mvnc.h>
#include "ncs_util.h"
#include "Ncs_Segmentation.hpp"

// graph file name
#define GRAPH_FILE_NAME "/dl/ros/seg_ncs_ros/src/seg_ncs/bag/graph"

//! image dimensions, network mean values for each channel in BGR order.
const int networkDim = 224;
//float networkMean[] = {71.60167789, 82.09696889, 72.30608881};
float networkMean[] = {100., 100., 100.};
const int target_h = 320;
const int target_w = 480;


namespace seg_ncs {

    Ncs_Segmentation::Ncs_Segmentation(ros::NodeHandle nh)
    : nodeHandle_(nh),
      imageTransport_(nodeHandle_){

        ROS_INFO("[SSD_Detector] Node started.");

        init_ncs();
        init();

    }

    Ncs_Segmentation::~Ncs_Segmentation() {

        //clear and close ncs
        ROS_INFO("Delete movidius SSD graph");
        retCode = mvncDeallocateGraph(graphHandle);
        graphHandle = NULL;

        free(graphFileBuf);
        retCode = mvncCloseDevice(deviceHandle);
        deviceHandle = NULL;
    }


    //  对movidius部分进行初始化
    void Ncs_Segmentation::init_ncs() {
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

    }

    // 对ros节点进行初始化
    void Ncs_Segmentation::init() {

        ROS_INFO("[Ncs_Segmentation] init().");

        //todo: read parameters from yaml
        imageSubscriber_ = imageTransport_.subscribe("/camera/image", 1,
                                   &Ncs_Segmentation::imageCallback, this);
        imageSegPub_ = imageTransport_.advertise("/camera/seg_out", 1);

    }


    //对推理结果进行展示
    cv::Mat Ncs_Segmentation::ncs_result_process(float* output, int h, int w)
    {

        //    out = out.reshape(-1, 2).T.reshape(2, 331, -1)
        //    out = out.argmax(axis=0)
        //    out = out[:-11, :-11]
        //    for (int i = 20000; i < 20050 ; ++i) {
        //        printf("ncs out: %f", output[i]);
        //    }

        // the output of graph was bigger than original image
        int margin = 11;
        int h_margin = h + margin;
        int w_margin = w + margin;

        cv::Mat mask_gray(h, w, CV_8UC1);
        cv::Mat mask;

        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                if(output[2*(w_margin*i + j)] < output[2*(w_margin*i + j) + 1]){
                    mask_gray.at<uchar>(i,j) = 255;
                } else{
                    mask_gray.at<uchar>(i,j) = 0;
                }
            }
        }
        // 灰度图转为彩色图
        cv::cvtColor(mask_gray, mask, cv::COLOR_GRAY2BGR);

        return mask;

    }


    // callback for inference
    void Ncs_Segmentation::imageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        ROS_INFO("callback");
        try
        {
            //resize，为了后面的融合展示
            cv::Mat ROS_img = cv_bridge::toCvShare(msg, "bgr8")->image;
            cv::Mat ROS_img_resized;
            cv::resize(ROS_img, ROS_img_resized, cv::Size(480,320), 0, 0, CV_INTER_LINEAR);

//            cv::imshow("ori", ROS_img_resized);

            // 将cvmat转为movidius使用的image类型
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
                    cv::Mat mask = ncs_result_process(resultData32, target_h, target_w);

                    free(resultData32);
                    delete imageBufFp16;

                    //图像混合
                    double alpha = 0.7;
                    cv::Mat out_img;
                    cv::addWeighted(ROS_img_resized, alpha, mask, 1 - alpha, 0.0, out_img);
                    // 发布topic
                    sensor_msgs::ImagePtr msg_seg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", out_img).toImageMsg();
                    imageSegPub_.publish(msg_seg);
                    ros::spinOnce();
                }
            }

        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
        }
        cv::waitKey(10);
    }

}
