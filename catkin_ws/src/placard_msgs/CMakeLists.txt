cmake_minimum_required(VERSION 2.8.3)
project(placard_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  roscpp
  rospy
  sensor_msgs
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  RegionProposalImage.msg
)
## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package(

)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

