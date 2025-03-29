import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
from torch import nn
from torchvision import transforms, models
import cv2
import torch.nn.functional as F
from std_msgs.msg import String

class ImageClassificationNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        # Define the correspondence between class names and indexes
        self.idx_to_class = {0: 'barefoot', 1: 'others', 2: 'shoes', 3: 'socks'}

        # Select device (use CUDA if GPU is available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f'Using device: {self.device}')

        # Load Model
        self.model = self.load_model('./final_weight.pth')
        
        self.publisher_ = self.create_publisher(String, '/inference_result', 10)

        # Subscribe to ROS Image messages
        self.subscription = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10
        )
        self.bridge = CvBridge()
        self.get_logger().info('ImageClassificationNode initialized.')

        # A list to store the inference results
        self.results = []
        
    def publish_inference_result(self, str):
        inference_result = str
        msg = String()
        msg.data = inference_result
        self.publisher_.publish(msg)

    def load_model(self, model_path):
        """
        Load the model and configure it for inference.
        """
        self.get_logger().info('Loading model...')
        # Prepare ResNet18 model
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, len(self.idx_to_class))  # Adjust the final layer
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)  # Transfer the model to your device
        model.eval()  # Set to inference mode
        self.get_logger().info('Model loaded successfully.')
        return model

    def image_callback(self, msg):
        """
        Receives images and performs inference using the model.
        """
        try:
            # Convert ROS Image message to OpenCV format image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Pretreatment
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((100, 100)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            input_tensor = transform(cv_image).unsqueeze(0).to(self.device)

            # inference
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)  # Calculate the confidence
            confidence, predicted = probabilities.max(1)
            class_name = self.idx_to_class[predicted.item()]

            # Save the inference results
            self.results.append((class_name, confidence.item()))

            # After processing 10 images, the most reliable result is output.
            if len(self.results) == 10:
                best_result = max(self.results, key=lambda x: x[1])
                self.publish_inference_result(best_result[0])
                self.get_logger().info(f'Most confident prediction: {best_result[0]} (Confidence: {best_result[1]:.4f})')
                self.results = []  # Reset

        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ImageClassificationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


