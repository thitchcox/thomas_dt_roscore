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
        #self.pub_lane_img = rospy.Publisher("~lane_img", Image, queue_size=1)
        self.pub_lane_pose = rospy.Publisher("~lane_pose", LanePose, queue_size=1)

        # Subscriptions
        self.subSeglistGroundProjection = rospy.Subscriber("~seglist_ground_projection",SegmentList, self.updateEstimate, queue_size=1)
        #self.sub_fsm_mode = rospy.Subscriber("~fsm_mode", FSMState, self.cbMode, queue_size=1)

        # safe shutdown
        rospy.on_shutdown(self.custom_shutdown)
        # ##################################
        # ######## WORKING VARIABLES #######
        # ##################################

        self.testCount = 0
        # ##################################

    # ##########################################################################
    # ######################## MAIN ESTIMATOR FUNCTION ####################### #
    # ##########################################################################
    def updateEstimate(self, segListMsg):
        whitePointsList = []
        yellowPointsList = []
        laneWidth = 0.2
        
        self.testCount = self.testCount + 1

        # Extract data from segments back into separated arrays.
        for segment in segListMsg.segments:
            if segment.color == segment.WHITE:
                whitePointsList.append(np.array([segment.points[0].x,segment.points[0].y]))
            if segment.color == segment.YELLOW:
                yellowPointsList.append(np.array([segment.points[0].x,segment.points[0].y]))
        whitePointsArray = np.array(whitePointsList)
        yellowPointsArray = np.array(yellowPointsList)
        
        # TODO - Treat cases where there are no segments!
        # Fit straight lines to data
        wShape = np.shape(whitePointsArray)
        yShape = np.shape(yellowPointsArray)
        nWhite = wShape[0]
        nYellow = yShape[0]
        phiWhite = 0
        phiYellow = 0
        distWhite = 0
        distYellow = 0
        if nWhite >= 2: # Need minimum of two points to define a line.
            zWhite = np.polyfit(whitePointsArray[:,0],whitePointsArray[:,1],deg=1) # Least-squares fit straight line
            b = -1
            m = zWhite[0]
            c = zWhite[1]
            x = -m*c/(m**2 + b**2)
            y = -b*c/(m**2 + b**2)
            #tVec = np.array([1,m])
            #nVec = np.array([1,-1/m])
            #phi = np.arccos(np.dot(tVec,np.array([1,0])/np.linalg.norm(tVec)))
            #distWhite = np.sqrt(x**2 + y**2)  # Calculate shortest distance: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
            distWhite = m*0.1 + c
            phiWhite = np.arcsin(-m/(np.sqrt(1 + m**2)))
            
        if nYellow >= 2:  # Need minimum of two points to define a line.
            zYellow = np.polyfit(yellowPointsArray[:,0],yellowPointsArray[:,1],deg=1)
            b = -1
            m = zYellow[0]
            c = zYellow[1]
            x = -m*c/(m**2 + b**2)
            y = -b*c/(m**2 + b**2)
            #tVec = np.array([1,m])
            #nVec = np.array([1,-1/m])
            #phi = np.arccos(np.dot(tVec,np.array([1,0])/np.linalg.norm(tVec)))
            phiYellow = np.arcsin(-m/(np.sqrt(1 + m**2)))
            #distYellow = np.sqrt(x**2 + y**2)
            distYellow = m*0.1 + c
        if nWhite >= 2 and nYellow >= 2:
            phi = (phiWhite + phiYellow)/2
            d =  (distWhite + distYellow)/2
        elif nWhite >= 2:
            phi = phiWhite - 0.6
            d = distWhite + (-laneWidth/2)
        elif nYellow >= 2:
            phi = phiYellow + 0.6
            d = distYellow + (laneWidth/2)
        else:
            phi = 0
            d = 0
        print(whitePointsArray)
        print("Iter: %d" % self.testCount," nWhite: %.2f" % nWhite, " nYellow: %.2f" % nYellow," phi: %.2f" % phi," distWhite: %.2f" % distWhite," distYellow: %.2f" % distYellow," d %.2f" % d) 
       
        #" phiWhite: %.2f" % phiWhite," phiYellow: %.2f" % phiYellow,
        """
        print(whitePointsArray)
        print(zWhite)
        print(whitePointsArray[:,0])
        print(whitePointsArray[:,1])
        """
         # build lane pose message to send
        lanePose = LanePose()
        lanePose.header.stamp = segListMsg.header.stamp
        lanePose.d = d
        lanePose.phi = phi
        lanePose.in_lane = True
        # XXX: is it always NORMAL?
        lanePose.status = lanePose.NORMAL

        self.pub_lane_pose.publish(lanePose)

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