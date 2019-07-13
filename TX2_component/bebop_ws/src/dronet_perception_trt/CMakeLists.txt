cmake_minimum_required(VERSION 2.8.3)
project(dronet_perception_trt)

set (CMAKE_CXX_STANDARD 11)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  std_msgs
  message_generation
  catkin_simple REQUIRED
)

find_package(OpenCV REQUIRED)

add_message_files(
  FILES
  CNN_out.msg
)

catkin_simple()

########################

#####
# TensorRT lib is compiled seperately, but is tightly coupled with
# the NN architecture and input/output size that is being used by ros.
# This means it should probably be recompiled, and its code should change,
# for each new ros project.
#####

add_library(trtinference SHARED IMPORTED)
set_property(TARGET trtinference PROPERTY IMPORTED_LOCATION "/home/nvidia/tensorRT/lib/libtrtinference.so")

set(trtinference_INCLUDE_DIRS
          "/home/nvidia/tensorRT/trtinference/")
########################


###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${trtinference_INCLUDE_DIRS}
)


cs_add_executable(dronet_perception_trt src/dronet.cpp)

target_link_libraries(dronet_perception_trt trtinference ${OpenCV_LIBRARIES})

cs_export()
