#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <stdio.h>
#include <stdlib.h>
#include <mvnc.h>
#include "ncs_util.h"
#include "Ncs_Segmentation.hpp"


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

        // load param
        std::string graphPath;
        std::string graphModel;
        bool flip_flag;
        nodeHandle_.param("seg_inception/graph_file/name", graphModel, std::string("seg_ncs_inception_graph"));
        nodeHandle_.param("graph_path", graphPath, std::string("graph"));
        graphPath += "/" + graphModel;
        GRAPH_FILE_NAME = new char[graphPath.length() + 1];
        strcpy(GRAPH_FILE_NAME, graphPath.c_str());
        nodeHandle_.param("seg_inception/networkDim", networkDim, 224);
        nodeHandle_.param("seg_inception/target_h", target_h, 320);
        nodeHandle_.param("seg_inception/target_w", target_w, 480);
        nodeHandle_.param("camera/image_flip", flip_flag, false);

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

        // load param
        std::string cameraTopicName;
        std::string segTopicName;
        nodeHandle_.param("subscribers/camera_reading/topic", cameraTopicName,
                            std::string("/camera/image"));
        nodeHandle_.param("subscribers/seg_image/topic", segTopicName,
                          std::string("/seg_ros/seg_image"));

        imageSubscriber_ = imageTransport_.subscribe(cameraTopicName, 1,
                                   &Ncs_Segmentation::imageCallback, this);
        imageSegPub_ = imageTransport_.advertise(segTopicName, 1);

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
    void Ncs_Segmentation::imageCallback(const sensor_msgs::ImageConstPtr &msg)
    {
//        ROS_DEBUG("[Seg:callback] image received.");
        cv_bridge::CvImagePtr cam_image;

        try {
            cam_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception &e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        if(cam_image){
            //flip
            cv::Mat image0 = cam_image->image.clone();
            IplImage copy = image0;
            IplImage *frame = &copy;
            //std::cout << "flipFlag: " << flipFlag << std::endl;
            if(flipFlag)
                cvFlip(frame, NULL, 0); //翻转
            camImageCopy_ = cv::cvarrToMat(frame, true);
        }

        //resize，为了后面的融合展示
//        cv::Mat ROS_img = cv_bridge::toCvShare(msg, "bgr8")->image;
        cv::Mat ROS_img_resized;
        cv::resize(camImageCopy_, ROS_img_resized, cv::Size(480, 320), 0, 0, CV_INTER_LINEAR);

        // 将cvmat转为movidius使用的image类型
        unsigned char *img = cvMat_to_charImg(camImageCopy_);
        unsigned int graphFileLen;
        half *imageBufFp16 = LoadImage(img, target_w, target_h, camImageCopy_.cols, camImageCopy_.rows, networkMean);

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
            }
        }
        return;
    }

}
