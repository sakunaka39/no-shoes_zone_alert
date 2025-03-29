import launch
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_inference_package',
            executable='inference_node.py',
            output='screen',
        ),
    ])

