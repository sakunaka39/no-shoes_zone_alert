# src/my_inference_package/launch/multi_node_launch.py

import launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Run urg_node (set log level to WARN)
        Node(
            package='urg_node',
            executable='urg_node_driver',
            name='urg_node_driver',
            output='log',
            #parameters=[{'params_file': '/home/jetson/jikken3_vision_recognition/ros2_ws/src/urg_node/config/urg_node_params.yaml'}],
            arguments=['--ros-args', '--log-level', 'WARN']
        ),
        
        # Run my_laser_processor (log level set to WARN)
        Node(
            package='my_laser_processor',
            executable='laser_processor',
            name='laser_processor',
            output='log',
            arguments=['--ros-args', '--log-level', 'WARN']
        ),
        
        # Run my_camera_package (set log level to WARN)
        Node(
            package='my_camera_package',
            executable='camera_capture',
            name='camera_capture',
            output='log',
            arguments=['--ros-args', '--log-level', 'WARN']
        ),
        
        # Run my_inference_package (log level is default)
        Node(
            package='my_inference_package',
            executable='inference_node',
            name='inference_node',
            output='log',
            arguments=['--ros-args', '--log-level', 'WARN']
        ),
        
        # Running my_control_package
        Node(
            package='my_control_package',
            executable='control_node',
            name='control_node',
            output='screen',
            arguments=['--ros-args', '--log-level', 'WARN']
        ),
    ])
