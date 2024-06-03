#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<rclcpp::Node>("publisher_node");

    auto publisher = node->create_publisher<std_msgs::msg::Int32>("Command", 10);

    auto message = std_msgs::msg::Int32();
    message.data = 98;

    // Kirim pesan
    RCLCPP_INFO(node->get_logger(), "Publishing: '%d'", message.data);
    publisher->publish(message);

    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}
