cmake_minimum_required(VERSION 2.8.3)
project(placard_detection)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  placard_msgs
)

catkin_package(

)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

