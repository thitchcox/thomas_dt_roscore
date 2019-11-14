#!/usr/bin/env python
import math
import rospy
# Import the different message structures that we will use.
from duckietown_msgs.msg import Segment, SegmentList, Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped
import time
import numpy as np

class pure_pursuit(object):
    # ##########################################################################
    # ############################## CONSTRUCTOR ############################# #
    # ##########################################################################
    def __init__(self):
        np.set_printoptions(precision=2)
        self.node_name = rospy.get_name()

        # Publications
        self.pub_car_cmd = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1)
        self.pub_radius_limit = rospy.Publisher("~radius_limit", BoolStamped, queue_size=1)
        self.pub_actuator_limits_received = rospy.Publisher("~actuator_limits_received", BoolStamped, queue_size=1)

        # Subscriptions
       # self.sub_lane_reading = rospy.Subscriber("~lane_pose", LanePose, self.PoseHandling, "lane_filter", queue_size=1) # Need to build callback. Might not be necessary
        self.sub_wheels_cmd_executed = rospy.Subscriber("~wheels_cmd_executed", WheelsCmdStamped, self.updateWheelsCmdExecuted, queue_size=1) # Need to build callback
        self.sub_actuator_limits = rospy.Subscriber("~actuator_limits", Twist2DStamped, self.updateActuatorLimits, queue_size=1) # Need to build callback
        self.pub_seglist_filtered = rospy.Subscriber("~seglist_filtered",SegmentList, self.updateCommands, queue_size=1)
        # safe shutdown
        rospy.on_shutdown(self.custom_shutdown)


    # ##########################################################################
    # ######################### MAIN CONTROL FUNCTION ######################## #
    # ##########################################################################
    def updateCommands(self, segListMsg):
        v_bar = 0.3
        omega = 0
        projDist = 0.25 # Distance to project forward along tangent
        lanewi
    # if a safe shutdown is necessarydth = 0.7

        whiteAvgX1 = 0
        whiteAvgY1 = 0
        yellowAvgX1 = 0
        yellowAvgY1 = 0
        whiteAvgX2 = 0
        whiteAvgY2 = 0
        yellowAvgX2 = 0
        yellowAvgY2 = 0
        nYellow = 0
        nWhite = 0
        segAvgWhite = Segment()
        segAvgYellow = Segment()
        segAvg = Segment()
        # Take an average of all the white and yellow segments.
        # Warning: Direction of the points matters...possible to get two segments to "cancel out" when averaging
        for segment in segListMsg.segments:
            # Check direction. To make sure segments are always points ``away'' from vehicle.
            dist1 = math.sqrt(segment.points[0].x ** 2 + segment.points[0].y ** 2)
            dist2 = math.sqrt(segment.points[1].x ** 2 + segment.points[1].y ** 2)
            if dist2 >= dist1:
                i1 = 0
                i2 = 1
            else:
                i1 = 1
                i2 = 0

            if (segment.color == segment.WHITE): 
                # Average the white segments together
                whiteAvgX1 = whiteAvgX1 + segment.points[i1].x
                whiteAvgY1 = whiteAvgY1 + segment.points[i1].y
                whiteAvgX2 = whiteAvgX2 + segment.points[i2].x
                whiteAvgY2 = whiteAvgY2 + segment.points[i2].y
                nWhite = nWhite + 1
            elif (segment.color == segment.YELLOW):
                # Average the yellow segments together
                yellowAvgX1 = yellowAvgX1 + segment.points[i1].x
                yellowAvgY1 = yellowAvgY1 + segment.points[i1].y
                yellowAvgX2 = yellowAvgX2 + segment.points[i2].x
                yellowAvgY2 = yellowAvgY2 + segment.points[i2].y
                nYellow = nYellow + 1
        if nWhite != 0:
            segAvgWhite.points[0].x = whiteAvgX1/nWhite
            segAvgWhite.points[0].y = whiteAvgY1/nWhite
            segAvgWhite.points[1].x = whiteAvgX2/nWhite
            segAvgWhite.points[1].y = whiteAvgY2/nWhite
        if nYellow != 0:
            segAvgYellow.points[0].x = yellowAvgX1/nYellow
            segAvgYellow.points[0].y = yellowAvgY1/nYellow
            segAvgYellow.points[1].x = yellowAvgX2/nYellow
            segAvgYellow.points[1].y = yellowAvgY2/nYellow 

        # Now that we have the ``average'' white and yellow segments, find overall average segment.
        if nWhite != 0 and nYellow != 0:
            segAvg.points[0].x = (segAvgWhite.points[0].x + segAvgYellow.points[0].x)/2
            segAvg.points[0].y = (segAvgWhite.points[0].y + segAvgYellow.points[0].y)/2
            segAvg.points[1].x = (segAvgWhite.points[1].x + segAvgYellow.points[1].x)/2 
            segAvg.points[1].y = (segAvgWhite.points[1].y + segAvgYellow.points[1].y)/2
        elif nWhite != 0 and nYellow == 0:
            # White lines only
            # Offset to the left by laneWidth/2
            segAvg = segAvgWhite
            segAvg.points[0].y = segAvg.points[0].y + lanewidth/2
            segAvg.points[1].y = segAvg.points[1].y + lanewidth/2
        elif nWhite == 0 and nYellow != 0:
            # Yellow lines only
            # Offset to the right by laneWidth/2
            segAvg = segAvgYellow
            segAvg.points[0].y = segAvg.points[0].y - lanewidth/2
            segAvg.points[1].y = segAvg.points[1].y - lanewidth/2
            
        # Calculate midpoint direction vector of overall average segment
        segPos = np.array([((segAvg.points[0].x + segAvg.points[1].x)/2),((segAvg.points[0].y + segAvg.points[1].y)/2)])
        segVec = np.array([segAvg.points[1].x,segAvg.points[1].y]) - np.array([segAvg.points[0].x,segAvg.points[0].y])
        mag = np.linalg.norm(segVec)
        if mag != 0:
            # Normalize direction vector of segment
            segVec = segVec/mag

        # Project forward to create desired point
        desPos = segPos + segVec * projDist

        # If lines are detected, calculate pure pursuit control effort. Otherwise, go straight.
        if (nWhite + nYellow)!= 0:
            L = np.linalg.norm(desPos)
            sinAlpha = desPos[1]
            omega = 3*2*v_bar*sinAlpha/L
        else:
            L = 0
            omega = 0

        # Publish the command
        car_control_msg = Twist2DStamped()
        car_control_msg.v = v_bar
        car_control_msg.omega = omega
        self.pub_car_cmd.publish(car_control_msg)
        print("Angular Velocity: %.2f" % omega,"L: %.2f" % L, "nY: %d" % nYellow, "nW: %d" % nWhite, "segPos: [%.2f , %.2f]" % (segPos[0], segPos[1]), "segVec: [%.2f , %.2f]" % (segVec[0], segVec[1])) 








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
    rospy.init_node('pure_pursuit', anonymous=False)
    pure_pursuit_node = pure_pursuit()
    rospy.spin()  