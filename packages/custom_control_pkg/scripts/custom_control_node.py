#!/usr/bin/env python
import math
import rospy
# Import the different message structures that we will use.
from duckietown_msgs.msg import Segment, SegmentList, Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped
import time
import numpy as np

class custom_control(object):
    # ##########################################################################
    # ############################## CONSTRUCTOR ############################# #
    # ##########################################################################
    def __init__(self):
        np.set_printoptions(precision=2)
        self.node_name = rospy.get_name()

        # Publications
        self.pub_car_cmd = rospy.Publisher("/default/lane_controller_node/car_cmd", Twist2DStamped, queue_size=1)
        self.pub_radius_limit = rospy.Publisher("~radius_limit", BoolStamped, queue_size=1)
        self.pub_actuator_limits_received = rospy.Publisher("~actuator_limits_received", BoolStamped, queue_size=1)

        # Subscriptions
        self.sub_lane_reading = rospy.Subscriber("~lane_pose", LanePose, self.updateCommands, queue_size=1)
        self.sub_wheels_cmd_executed = rospy.Subscriber("~wheels_cmd_executed", WheelsCmdStamped, self.updateWheelsCmdExecuted, queue_size=1) # Need to build callback
        self.sub_actuator_limits = rospy.Subscriber("~actuator_limits", Twist2DStamped, self.updateActuatorLimits, queue_size=1) # Need to build callback
        #self.pub_seglist_filtered = rospy.Subscriber("~seglist_filtered",SegmentList, self.updateCommands, queue_size=1)
        # safe shutdown
        rospy.on_shutdown(self.custom_shutdown)


    # ##########################################################################
    # ######################### MAIN CONTROL FUNCTION ######################## #
    # ##########################################################################
    def updateCommands(self, lanePoseMsg):

        d = lanePoseMsg.d
        phi = lanePoseMsg.phi

        
        v_bar = 0.6*max(1 - min(abs(phi)/(3.14159/2),1),0.01)

        k_phi = 1
        k_d = k_phi ** 2/(4*v_bar)


        u = - (k_d*d + k_phi*phi)
        

        # Publish the command
        car_control_msg = Twist2DStamped()
        car_control_msg.v = v_bar
        car_control_msg.omega = u
        self.pub_car_cmd.publish(car_control_msg)
        print("Angular Velocity: %.2f" % u," d: %.2f" % d, " phi: %.2f" % phi," v_bar: %.2f" % v_bar) 








    # ##########################################################################
    # ######################## HOUSEKEEPING FUNCTIONS ######################## #
    # ##########################################################################
    def updateWheelsCmdExecuted(self, msg_wheels_cmd):
        self.wheels_cmd_executed = msg_wheels_cmd

    def updateActuatorLimits(self, msg_actuator_limits):
        self.actuator_limits = msg_actuator_limits
        rospy.logdebug("actuator limits updated to: ")
        rospy.logdebug("actuator_limits.v: " + str(self.actuator_limits.v))
        rospy.logdebug("actuator_limits.omega: " + str(self.actuator_limits.omega))
        msg_actuator_limits_received = BoolStamped()
        msg_actuator_limits_received.data = True
        self.pub_actuator_limits_received.publish(msg_actuator_limits_received)


    # ############################# CUSTOM SHUTDOWN ############################
    def custom_shutdown(self):
        rospy.loginfo("[%s] Shutting down..." % self.node_name)

        # Stop listening
        self.sub_wheels_cmd_executed.unregister()
        self.sub_actuator_limits.unregister()


        # Send stop command
        car_control_msg = Twist2DStamped()
        car_control_msg.v = 0.0
        car_control_msg.omega = 0.0
        self.pub_car_cmd.publish(car_control_msg)

        rospy.sleep(0.5)    #To make sure that it gets published.
        rospy.loginfo("[%s] Shutdown" %self.node_name)


if __name__ == '__main__':
    rospy.init_node('custom_control', anonymous=False)
    custom_control_node = custom_control()
    rospy.spin()  