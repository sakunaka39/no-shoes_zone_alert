#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <thread>
#include <iostream>

class CameraCapture : public rclcpp::Node
{
public:
    CameraCapture() : Node("camera_capture"), capturing_(false)
    {
        bool_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/laser_bool", 10, std::bind(&CameraCapture::bool_callback, this, std::placeholders::_1));

        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/image", 10);

        cap_.open(0);
        if (!cap_.isOpened())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to open camera");
            rclcpp::shutdown();
        }
    }

private:
    void bool_callback(const std_msgs::msg::Bool::SharedPtr msg)
    {
        if (msg->data && !capturing_)
        {
            capturing_ = true;
            std::thread(&CameraCapture::capture_images, this).detach();
        }
    }

    void capture_images()
    {
        for (int i = 0; i < 10; ++i)
        {
            cv::Mat frame;
            cap_ >> frame;

            if (frame.empty())
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to capture frame");
                continue;
            }

            // Crop to squares
            int size = std::min(frame.cols, frame.rows);
            cv::Rect roi((frame.cols - size) / 2, (frame.rows - size) / 2, size, size);
            cv::Mat square_frame = frame(roi);

            // Publish to topic
            auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", square_frame).toImageMsg();
            image_pub_->publish(*msg);

            RCLCPP_INFO(this->get_logger(), "Published image %d", i);
            std::this_thread::sleep_for(std::chrono::milliseconds(33)); // 30fps = 33ms
        }

        RCLCPP_INFO(this->get_logger(), "Capture session complete.");
        std::this_thread::sleep_for(std::chrono::milliseconds(1500)); // Interval 1.5 seconds
        capturing_ = false;
    }

    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr bool_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    cv::VideoCapture cap_;
    bool capturing_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraCapture>());
    rclcpp::shutdown();
    return 0;
}


