cmake_minimum_required(VERSION 3.8)
project(my_camera_package)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

add_executable(camera_capture src/camera_capture.cpp)
ament_target_dependencies(camera_capture
  rclcpp
  sensor_msgs
  cv_bridge
)

install(TARGETS camera_capture
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

