#pragma once

#include <stdint.h>

#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Empty.h>
#include <sensor_msgs/Image.h>
#include "dronet_perception_trt/CNN_out.h"

// Pick either image stream (for testing) or video stream from bebop
#define USE_REFERENCE_IMAGE_STREAM
#ifndef USE_REFERENCE_IMAGE_STREAM
#define USE_BEBOP_STREAM
#endif

#ifdef USE_REFERENCE_IMAGE_STREAM
#define IMAGES_PATH "/home/nvidia/dronet/datasets/fake_dataset2/fake2/images/"
//#define IMAGES_PATH "/home/nvidia/dronet/datasets/fake_dataset/fake1/images/"
//#define IMAGES_PATH "/home/nvidia/dronet/datasets/BAG3_test/images/"
//#define IMAGES_PATH "/home/nvidia/dronet/datasets/collision_dataset/testing/DSCN2571/images/"
#endif

static uint16_t default_resize_values[2] = {320 , 240};
static uint8_t default_crop_values[2] = {200, 200};

namespace dronet
{

#define MAX_SPINS (512000 * 4 * 20)

class dronetPerception final
{

public:
  dronetPerception(const std::string uff_path_,
                   const std::string imgs_rootpath,
                   const uint16_t traget_size[2],
                   const uint8_t crop_size[2],
                   const ros::NodeHandle& nh = ros::NodeHandle(),
                   const ros::NodeHandle& nh_private = ros::NodeHandle("~"));
  dronetPerception() : dronetPerception("/home/nvidia/bebop_ws/src/dronet_perception_trt/models/model_tensorrt.uff",
                                        "/undefined",
                                        default_resize_values,
                                        default_crop_values,
                                        ros::NodeHandle(),
                                        ros::NodeHandle("~")) {}

  
  void run();

private:

  // ROS 
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  ros::Subscriber feedthrough_sub_;
  ros::Subscriber land_sub_;
  ros::Subscriber image_sub_;
  ros::Publisher predict_pub_;

  // Callback
  void callback_feedthrough(const std_msgs::Bool& msg);
  void callback_land(const std_msgs::Bool& msg);
  void callback_image(const sensor_msgs::ImageConstPtr& msg);

  // Parameters
  std::string json_model_path_;
  std::string uff_path_;
  uint16_t target_size_[2];
  uint8_t crop_size_[2];
  std::string imgs_rootpath_;
  std::string name_;

  // Internal variables
  bool use_network_out_;
  void * engine_;
  void * context_;

};



} // namespace dronet
