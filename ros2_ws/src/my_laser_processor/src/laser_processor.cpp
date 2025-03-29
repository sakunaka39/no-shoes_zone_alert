#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/bool.hpp>

class LaserProcessor : public rclcpp::Node
{
public:
    LaserProcessor() : Node("laser_processor")
    {
        // Subscriber: /scan
        laser_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&LaserProcessor::laser_callback, this, std::placeholders::_1));

        // Publisher: bool
        bool_pub_ = this->create_publisher<std_msgs::msg::Bool>("/laser_bool", 10);
    }

private:
    void laser_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        // Extract the necessary range of data to narrow the viewing angle
        float angle_min = -0.05; // Minimum angle
        float angle_max = 0.05;  // Maximum angle
        float angle_increment = msg->angle_increment;
        
        // Filtering laser_scan data based on angle range
        size_t start_idx = (angle_min - msg->angle_min) / angle_increment;
        size_t end_idx = (angle_max - msg->angle_min) / angle_increment;

        // Checking conditions (e.g., are there any obstacles within range?)
        bool object_detected = false;
        for (size_t i = start_idx; i <= end_idx; ++i)
        {
            if (msg->ranges[i] < 0.6) // When there is an obstacle within 0.6 meters
            {
                object_detected = true;
                break;
            }
        }

        // Publish the results
        auto bool_msg = std_msgs::msg::Bool();
        bool_msg.data = object_detected;
        bool_pub_->publish(bool_msg);

        RCLCPP_INFO(this->get_logger(), "Object detected: %s", object_detected ? "true" : "false");
    }

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_sub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr bool_pub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LaserProcessor>());
    rclcpp::shutdown();
    return 0;
}
