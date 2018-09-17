//
// Created by pesong on 18-9-5.
//

#ifndef SEG_NCS_ROS_NCS_SEGMENTATION_H
#define SEG_NCS_ROS_NCS_SEGMENTATION_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <thread>
#include <mvnc.h>
#include <boost/smart_ptr.hpp>
#include <boost/make_shared.hpp>
#include "ncs_util.h"
#include "Ncs_Segmentation.hpp"


namespace seg_ncs {

    // graph file name
    char *GRAPH_FILE_NAME;
    //! image dimensions, network mean values for each channel in BGR order.
    int networkDim;
    int target_h;
    int target_w;
    //cityscapes:
    float networkMean[] = {71.60167789, 82.09696889, 72.30608881};
    // float networkMean[] = {100., 100., 100.};

    class Ncs_Segmentation {
    public:

        /*!
        * Constructor.
        */
        explicit Ncs_Segmentation(ros::NodeHandle nh);

        /*!
         * Destructor.
         */
        ~Ncs_Segmentation();

    private:

        /*!
         * Initialize the movidius and ROS connections.
         */
        void init();
        void init_ncs();


        /*!
         * Callback of camera.
         * @param[in] msg image pointer.
         */
        void imageCallback(const sensor_msgs::ImageConstPtr& msg);

        cv::Mat ncs_result_process(float* output, int h, int w);

        cv::Mat camImageCopy_;
        std::thread segThread_;

        bool imageStatus_ = false;

        bool isNodeRunning_ = true;

        void *segThread();
        void *publishThread();

        void seg();

        cv::Mat getCVImage();

        bool getImageStatus(void);

        bool isNodeRunning(void);

        bool flipFlag;
        int demoDone_ = 0;
        cv::Mat seg_out_img;

        //! ROS node handle.
        ros::NodeHandle nodeHandle_;

        //! Advertise and subscribe to image topics.
        image_transport::ImageTransport imageTransport_;

        //! ROS subscriber and publisher.
        image_transport::Subscriber imageSubscriber_;
        image_transport::Publisher imageSegPub_;


    };
}
#endif //SEG_NCS_ROS_NCS_SEGMENTATION_H
