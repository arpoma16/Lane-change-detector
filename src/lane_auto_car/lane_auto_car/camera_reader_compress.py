#!/usr/bin/env python3
# Import the necessary libraries
import rclpy # Python Client Library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
import cv2 # OpenCV library
from sensor_msgs.msg import Image # Image is the message type
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images


class ImagePublisher(Node):

    def __init__(self):
        """
        Class constructor to set up the node
        """
        # Initiate the Node class's constructor and give it a name
        super().__init__('image_publisher')
        
        # Create the publisher. This publisher will publish an Image
        # to the video_frames topic. The queue size is 10 messages.
        self.publisher_ = self.create_publisher(Image, 'video_frame_compress', 10)
        
        # We will publish a message every 0.1 seconds
        timer_period = 0.1  # seconds
        
        # Create the timer
        self.timer = self.create_timer(timer_period, self.timer_callback)
            
        # Create a VideoCapture object
        # The argument '0' gets the default webcam.
        self.cap = cv2.VideoCapture('http://192.168.174.68:8080/video')
        # self.cap = cv2.VideoCapture(0)
            
        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

    def timer_callback(self):
        """
        Callback function.
        This function gets called every 0.1 seconds.
        """
        # Capture frame-by-frame
        # This method returns True/False as well
        # as the video frame.
        ret, frame = self.cap.read()
            
        if ret == True:
            # Publish the image.
            # The 'cv2_to_imgmsg' method converts an OpenCV
            # image to a ROS 2 image message
            _, compressed_image = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            msg = CompressedImage()
            msg.format = 'jpeg'
            msg.data = compressed_image.tostring()
            self.publisher_.publish(msg)
            self.get_logger().info('Imagen publicada')


def main(args = None):
    rclpy.init(args=args)
    node = ImagePublisher()
    rclpy.spin(node) # run the calbacks
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()   