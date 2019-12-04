#!/usr/bin/env python
import math
import rospy
# Import the different message structures that we will use.
from duckietown_msgs.msg import Segment, SegmentList, Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, Vector2D
import time
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from duckietown_utils.instantiate_utils import instantiate
from duckietown_utils.jpg import bgr_from_jpg
from geometry_msgs.msg import Point
from sensor_msgs.msg import CompressedImage, Image
from visualization_msgs.msg import Marker

from dt_project_lane_detector.timekeeper import TimeKeeper
import cv2
import threading
from dt_project_lane_detector.line_detector_plot import color_segment, drawLines




class custom_image(object):
    # ##########################################################################
    # ############################## CONSTRUCTOR ############################# #
    # ##########################################################################
    def __init__(self):
        # ##################################
        # ####### ROS SUBS AND PUBS ########
        # ##################################
        np.set_printoptions(precision=2)
        self.fsm_state = "unknown"
        self.node_name = rospy.get_name()

        # Publications

        # Subscriptions
        self.sub_image = rospy.Subscriber("~corrected_image/compressed", CompressedImage, self.processImage, queue_size=1)

        # safe shutdown
        rospy.on_shutdown(self.custom_shutdown)
        # ##################################
        # ######## WORKING VARIABLES #######
        # ##################################
        self.bridge = CvBridge()
        # ##################################


    # ##########################################################################
    # ############################## CONSTRUCTOR ############################# #
    # ##########################################################################
    def processImage(self, image_msg):

        # Decode from compressed image with OpenCV
        image_cv = bgr_from_jpg(image_msg.data)

        # Resize and crop image
        hei_original, wid_original = image_cv.shape[0:2]

        if self.image_size[0] != hei_original or self.image_size[1] != wid_original:
            # image_cv = cv2.GaussianBlur(image_cv, (5,5), 2)
            image_cv = cv2.resize(image_cv, (self.image_size[1], self.image_size[0]),
                                   interpolation=cv2.INTER_NEAREST)
        image_cv = image_cv[self.top_cutoff:,:,:]



        
    # ############################# CUSTOM SHUTDOWN ############################
    def custom_shutdown(self):
        rospy.loginfo("[%s] Shutting down..." % self.node_name)

        # Stop listening
        self.sub_image.unregister()

        rospy.sleep(0.5)    #To make sure that it gets published.
        rospy.loginfo("[%s] Shutdown" %self.node_name)


if __name__ == '__main__':
    rospy.init_node('custom_image', anonymous=False)
    custom_image_node = custom_image()
    rospy.spin()  