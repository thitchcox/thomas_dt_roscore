#!/usr/bin/env python
import math
import rospy
# Import the different message structures that we will use.
from duckietown_msgs.msg import Segment, SegmentList, Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState
import time
import numpy as np

class simple_estimator(object):
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

        #self.pub_car_cmd = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1)
        #self.pub_radius_limit = rospy.Publisher("~radius_limit", BoolStamped, queue_size=1)
        #self.pub_actuator_limits_received = rospy.Publisher("~actuator_limits_received", BoolStamped, queue_size=1)

        # Subscriptions
        #self.sub_lane_reading = rospy.Subscriber("~lane_pose", LanePose, self.updateCommands, queue_size=1)
        #self.sub_wheels_cmd_executed = rospy.Subscriber("~wheels_cmd_executed", WheelsCmdStamped, self.updateWheelsCmdExecuted, queue_size=1) # Need to build callback
        #self.sub_actuator_limits = rospy.Subscriber("~actuator_limits", Twist2DStamped, self.updateActuatorLimits, queue_size=1) # Need to build callback
        self.subSeglistGroundProjection = rospy.Subscriber("~seglist_ground_projection",SegmentList, self.updateEstimate, queue_size=1)
        #self.sub_fsm_mode = rospy.Subscriber("~fsm_mode", FSMState, self.cbMode, queue_size=1)

        # safe shutdown
        rospy.on_shutdown(self.custom_shutdown)
        # ##################################
        # ######## WORKING VARIABLES #######
        # ##################################
        self.x_reference = np.array([[4],[0.5]])
        self.theta = np.array([[-2 ** 2/(4*0.5)],[-2],[0]])
        self.previousTime = float(rospy.Time.now().to_sec()) # DOUBLE CHECK, might need to be changed.
        self.previousAdaptTime = float(rospy.Time.now().to_sec()) # DOUBLE CHECK, might need to be changed.
        self.x = np.array([[0],[0]])
        self.errorInt = 0
        self.testCount = 0
        # ##################################

    # ##########################################################################
    # ######################### MAIN CONTROL FUNCTION ######################## #
    # ##########################################################################
    def updateEstimate(self, segListMsg):
        print(segListMsg)
        print(self.testCount)
        self.testCount = self.testCount + 1



    # ##########################################################################
    # ######################## HOUSEKEEPING FUNCTIONS ######################## #
    # ##########################################################################



    # ############################# CUSTOM SHUTDOWN #s###########################
    def custom_shutdown(self):
        rospy.loginfo("[%s] Shutting down..." % self.node_name)

        # Stop listening
        self.subSeglistGroundProjection.unregister()

        rospy.sleep(0.5)    #To make sure that it gets published.
        rospy.loginfo("[%s] Shutdown" %self.node_name)


if __name__ == '__main__':
    rospy.init_node('simple_estimator', anonymous=False)
    simple_estimator_node = simple_estimator()
    rospy.spin()  