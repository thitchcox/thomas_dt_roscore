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
        self.theta = np.array([[-2 ** 2/(4*0.5)],[-2],[0]])
        self.previousTime = float(rospy.Time.now().to_sec()) # DOUBLE CHECK, might need to be changed.
        self.previousAdaptTime = float(rospy.Time.now().to_sec()) # DOUBLE CHECK, might need to be changed.
        self.x = np.array([[0],[0]])
        self.errorInt = 0
        # ##################################

    # ##########################################################################
    # ######################### MAIN CONTROL FUNCTION ######################## #
    # ##########################################################################
    def updateCommands(self, lanePoseMsg):

        d = lanePoseMsg.d
        phi = lanePoseMsg.phi
        self.x = np.array([[d],[phi]])
        T = float(rospy.Time.now().to_sec()) - float(self.previousTime) # TO BE CHANGED MAYBE
        u_command = 0
        u_max = 8
        
        # ######### SPEED CONTROL #########
        if abs(phi)> 1.0:
            v_bar = 0.0
        elif abs(phi)> 0.6:
            v_bar = 0.05
        elif abs(phi) > 0.3:
            v_bar = 0.1
        elif abs(phi) > 0.2:
            v_bar = 0.3
        elif abs(phi) <= 0.2:
            v_bar = 0.4
        else:
            v_bar = 0.4

        # ######## REFERENCE MODEL ######## (Could be saved in object properties)
        # Critically damped gains from class
        k_d = -1
        k_p = -k_d ** 2/(v_bar * 4)
        # Closed loop matrices
        A_r = np.array([[0,v_bar],[k_p, k_d]])
        B_r = np.array([[0],[1]])


        
        # Propagate reference state forward (Euler discretization)
        # Periodic reinitialization (presistency of excitation)
        if (float(rospy.Time.now().to_sec()) - self.previousAdaptTime) > 3:
            self.x_reference = self.x # Reset state
            self.previousAdaptTime = float(rospy.Time.now().to_sec()) 

        x_r_k = self.x_reference
        x_r_k1 = np.matmul((np.identity(2) + T*A_r),x_r_k)
        

        # ########## ADAPTIVE LAW #########
        if self.fsm_state == "LANE_FOLLOWING":
            gamma = 0.1 # adaptive gain
        else:
            gamma = 0
        x_k = self.x
        e_model = x_k - x_r_k
        psi = np.array([[0,0,0],[d,phi,u_command]]) # Basis matrix
        P = np.array([[0.25,-1],[-1,2]]) # Solution to Lyapunov equation
        theta_dot = -gamma*np.matmul(psi.transpose(), np.matmul(P , e_model)) # Adaptive law
        theta_k1 = self.theta + T*theta_dot # Integrate theta
        theta_k1[0] = max(min(theta_k1[0],0),-10) # Cap theta_k1 (anti-windup)
        theta_k1[1] = max(min(theta_k1[1],0),-10)

        # ############ CONTROL ############
        u_adaptive = self.theta[0]*x_k[0] + self.theta[1]*x_k[1]
        
        # PD Controller from class
        k_phi = -2
        k_d = -k_phi ** 2/(4*v_bar)
        u_PD = k_d*(x_k[0]-0.025) + k_phi*(x_k[1])

        # PID Controller
        kp = 10
        kd = 4
        ki = 0.3
        intMax = 1
        self.errorInt = max(min(self.errorInt,intMax),-intMax) # Anti windup
        u_PID = kp*(0.15 - x_k[0]) + kd*(0 - x_k[1]) + ki*self.errorInt
        if abs(u_PID) < u_max and self.fsm_state == "LANE_FOLLOWING": # Only increase integral if we arnt at the max control effort
            self.errorInt = self.errorInt - x_k[0]*T


        # Pure pursiut
        L = 0.1
        u_PP = 2*v_bar*x_k[0]/L


        # ########## PUBLISH #############
        #u = u_adaptive
        u = u_PD
        #u = u_PP
        #u = u_PID
        car_control_msg = Twist2DStamped()
        car_control_msg.v = v_bar
        car_control_msg.omega = u
        self.pub_car_cmd.publish(car_control_msg)

        # Update working variables
        self.previousTime = float(rospy.Time.now().to_sec()) # TO BE CHANGED MAYBE
        self.x_reference = x_r_k1
        self.theta = theta_k1

        # Print stuff
        #print(self.theta)
        #print(self.x_reference)
        
        #print(T)
        #print(self.x)
        #print(self.fsm_state)
        #print(self.errorInt)
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