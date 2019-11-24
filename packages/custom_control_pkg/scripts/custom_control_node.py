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
        self.x_reference = np.array([[0],[0]])
        self.theta = np.array([[-4 ** 2/(4*0.5)],[-4],[0]])
        self.previousTime = float(rospy.Time.now().to_sec()) # DOUBLE CHECK, might need to be changed.

        # ##################################

    # ##########################################################################
    # ######################### MAIN CONTROL FUNCTION ######################## #
    # ##########################################################################
    def updateCommands(self, lanePoseMsg):

        d = lanePoseMsg.d
        phi = lanePoseMsg.phi
        T = float(rospy.Time.now().to_sec()) - float(self.previousTime) # TO BE CHANGED MAYBE
        u_command = 0
        
        #v_bar = 0.6*max(1 - min(abs(phi)/(3.14159/2),1),0.01)
        v_bar = 0.5
        P = np.identity(2) # Consider moving to properties

        # ######## REFERENCE MODEL ######## (Could be saved in object properties)
        # Critically damped gains from class
        k_d = -4
        k_p = -k_d ** 2/(v_bar * 4)
        # Closed loop matrices
        A_r = np.array([[0,v_bar],[k_p, k_d]])
        B_r = np.array([[0],[1]])

        print(A_r)
        print(T)
        
        # Propagate reference state forward (Euler discretization)
        x_r_k = self.x_reference
        e_r_k = np.array([[d],[phi]]) - self.x_reference
        x_r_k1 = np.matmul((np.identity(2) + T*A_r),e_r_k)
        

        # ########## ADAPTIVE LAW #########
        if self.fsm_state == "LANE_FOLLOWING":
            gamma = 2 # adaptive gain
        else:
            gamma = 0
        x_k = np.array([[d],[phi]])
        e_model = x_k - x_r_k
        psi = np.array([[0,0,0],[d,phi,u_command]]) # Basis matrix
        theta_dot = -gamma*np.matmul(psi.transpose(), np.matmul(P , e_model))
        theta_k1 = self.theta + T*theta_dot

        # ############ CONTROL ############
        #k_phi = 1
        #k_d = k_phi ** 2/(4*v_bar)
        print(self.theta)
        print(x_r_k)
        print(self.fsm_state)
        u = self.theta[0]*x_k[0] + self.theta[1]*x_k[1]
        
        # Publish the command
        car_control_msg = Twist2DStamped()
        car_control_msg.v = v_bar
        car_control_msg.omega = u
        self.pub_car_cmd.publish(car_control_msg)

        # Update working variables
        self.previousTime = float(rospy.Time.now().to_sec()) # TO BE CHANGED MAYBE
        self.x_reference = x_r_k1
        self.theta = theta_k1
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