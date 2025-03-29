cmake_minimum_required(VERSION 3.8)
project(my_laser_processor)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(my_camera_package REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(laser_processor src/laser_processor.cpp)

ament_target_dependencies(laser_processor
  rclcpp
  my_camera_package
  sensor_msgs
)

install(TARGETS laser_processor
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

