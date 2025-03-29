import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import Jetson.GPIO as GPIO
import time
import os
from threading import Thread

# GPIO pin configuration
LED_PIN = 33
GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PIN, GPIO.OUT)

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        self.subscription = self.create_subscription(
            String,
            '/inference_result',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        #self.get_logger().info(f'Received inference result: {msg.data}')
        if (msg.data == "socks") or (msg.data == "barefoot"):
            self.activate_led_and_sound_ok()
        elif msg.data == "shoes":
            self.activate_led_and_sound_alert()

    def activate_led_and_sound_ok(self):
        try:
            os.system('aplay ./sound_ok.wav > /dev/null 2>&1')
        finally:
            GPIO.output(LED_PIN, GPIO.HIGH)
        self.get_logger().info('Welcome.')
            
    def activate_led_and_sound_alert(self):
        try:
            led_thread = Thread(target=self.blink_led)
            led_thread.start()
            os.system('aplay ./sound_alert.wav > /dev/null 2>&1')
            led_thread.join()
        finally:
            GPIO.output(LED_PIN, GPIO.HIGH)
        self.get_logger().info('Please take off your shoes.')

    def blink_led(self):
        for _ in range(5):
            GPIO.output(LED_PIN, GPIO.LOW)
            time.sleep(0.15)
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(0.15)

def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Control Node Stopped")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        GPIO.cleanup()

if __name__ == '__main__':
    main()

