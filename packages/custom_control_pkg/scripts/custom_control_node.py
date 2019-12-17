#!/usr/bin/env python
import math
import rospy
# Import the different message structures that we will use.
from duckietown_msgs.msg import Segment, SegmentList, Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState
import time
import numpy as np

class custom_control(object):
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
        self.pub_car_cmd = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1)
        self.pub_radius_limit = rospy.Publisher("~radius_limit", BoolStamped, queue_size=1)
        self.pub_actuator_limits_received = rospy.Publisher("~actuator_limits_received", BoolStamped, queue_size=1)

        # Subscriptions
        self.sub_lane_reading = rospy.Subscriber("~lane_pose", LanePose, self.updateCommands, queue_size=1)
        self.sub_wheels_cmd_executed = rospy.Subscriber("~wheels_cmd_executed", WheelsCmdStamped, self.updateWheelsCmdExecuted, queue_size=1) # Need to build callback
        self.sub_actuator_limits = rospy.Subscriber("~actuator_limits", Twist2DStamped, self.updateActuatorLimits, queue_size=1) # Need to build callback
        #self.pub_seglist_filtered = rospy.Subscriber("~seglist_filtered",SegmentList, self.updateCommands, queue_size=1)
        self.sub_fsm_mode = rospy.Subscriber("~fsm_mode", FSMState, self.cbMode, queue_size=1)

        # safe shutdown
        rospy.on_shutdown(self.custom_shutdown)
        # ##################################
        # ######## WORKING VARIABLES #######
        # ##################################
        self.x_reference = np.array([[4],[0.5]])
        self.previousTime = float(rospy.Time.now().to_sec()) # DOUBLE CHECK, might need to be changed.
        self.x = np.array([[0],[0]])
        self.errorInt = 0
        self.wheelSpan = 0.1
        self.x_hat = np.array([[0],[0],[0]])
        self.y = np.array([[0],[0]])
        self.K_obs = np.array([[0.2,0.3],[0,0.5],[0,0.02]])
        self.k_trim_actual = 0.2
        
        # ##################################

    # ##########################################################################
    # ######################### MAIN CONTROL FUNCTION ######################## #
    # ##########################################################################
    def updateCommands(self, lanePoseMsg):

        d = lanePoseMsg.d
        phi = lanePoseMsg.phi
        self.x = np.array([[d],[phi]])
        self.y = np.array([[d],[phi]])
        dt = float(rospy.Time.now().to_sec()) - float(self.previousTime) # TO BE CHANGED MAYBE
        u_command = 0
        u_max = 8
        v_bar = 0.2
        if abs(phi) > 1:
            v_bar = 0.2
        elif abs(phi) > 0.6:
            v_bar = 0.2

        # PD Controller from class
        k_phi = -2
        k_d = -k_phi ** 2/(4*v_bar)
        om_com = k_d*(self.x[0]) + k_phi*(self.x[1]) - v_bar/self.wheelSpan*self.x_hat[2]

        # Propagate Observer state
        A = np.array([[0,v_bar],[0,0]])
        B = np.array([[0],[1]])
        C = np.identity(2)
        L = np.array([[0],[v_bar/self.wheelSpan]])

        A_aug = np.vstack((np.hstack((A,L)),np.array([0,0,0])))
        B_aug = np.vstack((B,np.array([0])))
        C_aug = np.hstack((C,np.array([[0],[0]])))

        x_hat_dot = np.matmul(A_aug,self.x_hat) + B_aug*om_com + np.matmul(self.K_obs,self.y - np.matmul(C_aug,self.x_hat))
        if abs(d)< 0.2 and abs(phi) < 0.3:
            self.x_hat = self.x_hat + dt*x_hat_dot



        # ########## PUBLISH #############
        u = om_com + v_bar/self.wheelSpan*self.k_trim_actual
        car_control_msg = Twist2DStamped()
        car_control_msg.v = v_bar
        car_control_msg.omega = u
        self.pub_car_cmd.publish(car_control_msg)

        # Update working variables
        self.previousTime = float(rospy.Time.now().to_sec()) # TO BE CHANGED MAYBE

        # Print stuff
        #print(self.theta)
        #print(self.x_reference)
        
        #print(T)
        #print(self.x)
        #print(self.fsm_state)
        #print(self.errorInt)
        #print("Angular Velocity: %.2f" % u," d: %.2f" % d, " phi: %.2f" % phi," v_bar: %.2f" % v_bar) 
        print("Estimated Trim: %.2f" % self.x_hat[2]," Actual Trim: %.2f" % self.k_trim_actual," d: %.2f" % d, " phi: %.2f" % phi) 
       # print(self.x_hat)

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

    def cbMode(self,fsm_state_msg):
        self.fsm_state = fsm_state_msg.state    # String of current FSM state
        print "fsm_state changed in lane_controller_node to: " , self.fsm_state

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