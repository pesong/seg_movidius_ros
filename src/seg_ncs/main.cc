//
// Created by pesong on 18-9-5.
//
#include "Ncs_Segmentation.hpp"

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "seg_ncs");
    ros::NodeHandle nh, priv_nh("~");

    seg_ncs::Ncs_Segmentation ncs_segmentation(priv_nh);
    ros::spin();
}