#include <unistd.h>
#include <string>
#include <fstream>
#include <chrono>
#include <mutex>


#include <cv_bridge/cv_bridge.h>
#include "dronet_perception_trt/dronet.h"
#include "dronet_utils.h"
#include "trtinference.h"

#ifdef USE_REFERENCE_IMAGE_STREAM
#include <opencv2/highgui/highgui.hpp>
#include <dirent.h>
#include <algorithm> 
#endif

#define DEBUG_IMAGE_VALUES 1

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>


sensor_msgs::Image glob_img;
bool rdy = false;
unsigned int spins = 0;

namespace dronet_utils
{
float* crop_img(sensor_msgs::Image data, uint16_t target_size[2], uint8_t crop_size[2], cv::Mat **output_img)
{
  cv_bridge::CvImagePtr p_img = cv_bridge::toCvCopy(data, sensor_msgs::image_encodings::MONO8);
  cv::Mat *img = &(p_img->image);
  cv::Mat resized_img, cropped_img;
  cv::Mat* normalized_img = new cv::Mat(crop_size[0], crop_size[1], CV_32FC1, 11.0);
  
  cv::Size sz(target_size[0], target_size[1]);
  cv::Rect myROI(int( (target_size[0]-crop_size[0]) / 2),
                 int( (target_size[1]-crop_size[1])),
                 crop_size[0],
                 crop_size[1]);
  
  *output_img = normalized_img; // We need to delete normalized_img, but only after the image is used, so we return it to the calling function.
  
  // To do this efficiantly, we first resize, then crop (no copy), and finaly convert-copy only the cropped image to a new Mat.
  cv::resize(*img, resized_img, sz);
  cropped_img = resized_img(myROI);
  cropped_img.convertTo(*normalized_img, CV_32F, 1.0/255.0);
  
  *normalized_img = cv::Scalar(0.0);
#if DEBUG_IMAGE_VALUES
  float *ptr = (float*)normalized_img->data;
  printf("image1 content is: %d, %d, %d, %d\n", img->data[0], img->data[20], img->data[30], img->data[50]);
  printf("image2 content is: %d, %d, %d, %d\n", resized_img.data[0], resized_img.data[20], resized_img.data[30], resized_img.data[50]);
  printf("image3 content is: %d, %d, %d, %d\n", cropped_img.data[0], cropped_img.data[20], cropped_img.data[30], cropped_img.data[50]);
  printf("image4start content is: %f, %f, %f, %f\n", ptr[0], ptr[20], ptr[30], ptr[50]);
  printf("image4end content is: %f, %f, %f, %f\n", ptr[39950], ptr[39970], ptr[39980], ptr[39999]);
#endif
  
  return (float*)normalized_img->data;
}

#ifdef USE_REFERENCE_IMAGE_STREAM
std::vector<std::string> files;
std::vector<std::string>::iterator it;


float* read_next_img(cv::Mat **output_img)
{
  static bool first = true;
  
  if (first) {
    printf("first time, reading files for directory..\n");
    DIR * dir;
    dirent * pdir;
    dir = opendir(IMAGES_PATH); 
    while ((pdir = readdir(dir))) { 
        files.push_back(pdir->d_name); 
    }
    
    std::sort(files.begin(), files.end());
    
    it = files.begin();
    
    printf("file list is ready\n");
    printf("****************\n");
    first = false;
  }
  
  cv::Mat* normalized_img = new cv::Mat(200, 200, CV_32FC1, 11.0);
  cv::Mat resized_img, cropped_img;
  cv::Mat image;
  
  cv::Size sz(320,240);
  cv::Rect myROI(int( (320-200) / 2),
                 int( (240-200)),
                 200,
                 200);
  
  *output_img = normalized_img;
  
  while(it != files.end())
  {
      if ((*it).length() < 5) {
        it++;
        continue;
      }

      image = cv::imread((IMAGES_PATH + *it).c_str(), cv::IMREAD_GRAYSCALE);
      cv::resize(image, resized_img, sz);
      cropped_img = resized_img(myROI);
      cropped_img.convertTo(*normalized_img, CV_32F, 1.0/255.0);
      cv::imshow( "Display window", *normalized_img );
      cv::waitKey(1);
      
#if DEBUG_IMAGE_VALUES
        float *ptr = (float*)normalized_img->data;
        printf("image1 content is: %d, %d, %d, %d\n", image.data[0], image.data[20], image.data[30], image.data[50]);
        printf("image2 content is: %d, %d, %d, %d\n", resized_img.data[0], resized_img.data[20], resized_img.data[30], resized_img.data[50]);
        printf("image3 content is: %d, %d, %d, %d\n", cropped_img.data[0], cropped_img.data[20], cropped_img.data[30], cropped_img.data[50]);
        printf("image4start content is: %f, %f, %f, %f\n", ptr[0], ptr[20], ptr[30], ptr[50]);
        printf("image4end content is: %f, %f, %f, %f\n", ptr[39950], ptr[39970], ptr[39980], ptr[39999]);
#endif
      
      ++it;
      return (float*)normalized_img->data;
  }
  delete(normalized_img);
  return NULL;
}
#endif

} // namespace dronet_utils

namespace dronet
{


dronetPerception::dronetPerception(
    const std::string uff_path,
    const std::string imgs_rootpath, //TODO UNUSED
    const uint16_t target_size[2],
    const uint8_t crop_size[2],
    const ros::NodeHandle& nh,
    const ros::NodeHandle& nh_private)
  :   nh_(nh),
      nh_private_(nh_private),
      name_(nh_private.getNamespace())
{
  ROS_INFO("[%s]: Initializing Dronet Percepction", name_.c_str());
  
#ifdef USE_BEBOP_STREAM
  feedthrough_sub_ = nh_.subscribe("state_change", 1, &dronetPerception::callback_feedthrough, this);
  land_sub_ = nh_.subscribe("land", 1, &dronetPerception::callback_land, this);
  image_sub_ = nh_.subscribe("camera", 1, &dronetPerception::callback_image, this);
#endif

  predict_pub_ = nh_.advertise < dronet_perception_trt::CNN_out > ("cnn_predictions", 5);
  
  //load path and parameters
  uff_path_ = uff_path;
  target_size_[0] = target_size[0];
  target_size_[1] = target_size[1];
  crop_size_[0] = crop_size[0];
  crop_size_[1] = crop_size[1];
  
  use_network_out_ = false;
  
  
  ROS_INFO("NOW PREPARING TensorRT MODEL");
  int ret = trt_ros_inference::build_engine(uff_path_, crop_size_, &engine_, &context_);
  if (ret != 0) {
    ROS_INFO("trt engine build failed!!!");
    exit(-1);
  }
}

void dronetPerception::run()
{
  float outs[NUM_OF_OUTPUTS]; // NUM_OF_OUTPUTS is defined in trtinference.h
  float steer, coll;
  float *cropped_img;
  cv::Mat *to_clear = NULL;
  
  
 
  ROS_INFO("uff_path: %s", uff_path_.c_str());
  ROS_INFO("target_size_: %u,%u", target_size_[0], target_size_[1]);
  ROS_INFO("crop_size_: %u,%u", crop_size_[0], crop_size_[1]);
  ROS_INFO("use_network_out: %d", use_network_out_);

#ifdef TIMING_TEST
  auto tic = std::chrono::high_resolution_clock::now();
#endif

  while(ros::ok())
  {
    dronet_perception_trt::CNN_out msg;
    msg.header.stamp = ros::Time::now();
    
    // Get image from source
#ifdef USE_BEBOP_STREAM
    if ( rdy == false) {
      ros::spinOnce();  // Let ROS callbacks run
      if (spins++ > MAX_SPINS) {
        ROS_INFO("No incoming image callback for %u spins. aborting.", MAX_SPINS);
        goto err_exit;
      }
      continue;
    }
    rdy = false;  // This makes sure that we get a new image on the next iteration    
        
    cropped_img = dronet_utils::crop_img(glob_img, target_size_, crop_size_, &to_clear);
#else // USE_REFERENCE_IMAGE_STREAM
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
    cropped_img = dronet_utils::read_next_img(&to_clear);
    if (NULL == cropped_img)
      break;
#endif

    // Process image
    (void)trt_ros_inference::inference(engine_, context_, cropped_img, outs); // TODO add retval check
    delete(to_clear);
    
    // process output/send instructions 
    coll = outs[0];
    steer = outs[1];
    
    msg.steering_angle = steer;
    msg.collision_prob = coll;
    
    //ROS_INFO("=====================================");
    
    if (use_network_out_ == true)
    {
      ROS_INFO("Got predictions: steering=%05.2f, coll=%04.2f (Publishing commands!)", steer, coll);
      predict_pub_.publish(msg);
    }
    else
    {
      ROS_INFO("Got predictions: steering=%05.2f, coll=%04.2f (NOT publishing commands)", steer, coll);
      // TODO save img to root path here
    }
    //ROS_INFO("=====================================");
    
#ifdef TIMING_TEST
    auto toc = std::chrono::high_resolution_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(toc-tic).count())
    {
      tic = toc;
      ROS_INFO("1 seconds mark");
    }
#endif
  }

err_exit:
  // TODO LAND DRONE NOW
  // TODO KILL BEBOP_DRONE nodelet
  trt_ros_inference::delete_engine(engine_, context_);
  
  ROS_INFO("Dronet done!!!");
}

void dronetPerception::callback_image(const sensor_msgs::ImageConstPtr& msg)
{
  glob_img = *msg;
  rdy = true;
  spins = 0;
}

void dronetPerception::callback_feedthrough(const std_msgs::Bool& msg)
{
  use_network_out_ = msg.data;
}

void dronetPerception::callback_land(const std_msgs::Bool& msg)
{
  use_network_out_ = false;
}


} // namespace dronet



int main(int argc, char** argv)
{
    ros::init(argc, argv, "dronet");
    dronet::dronetPerception inference;

    inference.run();

    return 0;
}
