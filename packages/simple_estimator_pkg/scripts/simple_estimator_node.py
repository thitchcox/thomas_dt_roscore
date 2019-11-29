#!/usr/bin/env python
import math
import rospy
# Import the different message structures that we will use.
from duckietown_msgs.msg import Segment, SegmentList, Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, ColorRGBA #Imports msg
import time
import numpy as np
from sklearn import linear_model


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

        # Get vehicle name from namespace
        self.veh_name = rospy.get_namespace().strip("/")
        #rospy.loginfo("[%s] Vehicle name: %s" % (self.node_name, self.veh_name))

        # Publications
        self.pub_lane_pose = rospy.Publisher("~lane_pose", LanePose, queue_size=1)
        self.pub_lane = rospy.Publisher("~lane_markers",MarkerArray,queue_size=1)

        # Subscriptions
        self.subSeglistGroundProjection = rospy.Subscriber("~seglist_ground_projection",SegmentList, self.updateEstimate, queue_size=1)
        self.sub_velocity = rospy.Subscriber("~car_cmd", Twist2DStamped, self.predict)
        #self.sub_fsm_mode = rospy.Subscriber("~fsm_mode", FSMState, self.cbMode, queue_size=1)

        # safe shutdown
        rospy.on_shutdown(self.custom_shutdown)
        # ##################################
        # ######## WORKING VARIABLES #######
        # ##################################
        self.ransacW = linear_model.RANSACRegressor()
        self.ransacW.residual_threshold = 0.05
        self.ransacY = linear_model.RANSACRegressor()
        self.testCount = 0
        # ##################################

    # ##########################################################################
    # ######################## MAIN ESTIMATOR FUNCTION ####################### #
    # ##########################################################################
    def updateEstimate(self, segListMsg):
        whitePointsList = []
        yellowPointsList = []
        laneWidth = 0.25
        
        self.testCount = self.testCount + 1 # Just some counter, gives an idea of the speed.

        # Extract data from segments back into separated arrays.
        # ONLY IF THEY ARE CLOSE
        for segment in segListMsg.segments:
            if segment.color == segment.WHITE and segment.points[0].x <= 0.4:
                whitePointsList.append(np.array([segment.points[0].x,segment.points[0].y]))
            if segment.color == segment.YELLOW and segment.points[0].x <= 0.4:
                yellowPointsList.append(np.array([segment.points[0].x,segment.points[0].y]))
        whitePointsArray = np.array(whitePointsList)
        yellowPointsArray = np.array(yellowPointsList)
        
        # Fit straight lines to data
         # TODO - not immune to seeing the OTHER white line. Go through rigorously all the signs of the distances.
        wShape = np.shape(whitePointsArray)
        yShape = np.shape(yellowPointsArray)
        nWhite = wShape[0]
        nYellow = yShape[0]
        phiWhite = 0
        phiYellow = 0
        distWhite = 0
        distYellow = 0
        if nWhite >= 2: # Need minimum of two points to define a line.
            self.ransacW.fit(np.reshape(whitePointsArray[:,0],(-1,1)), np.array(np.reshape(whitePointsArray[:,1],(-1,1))))
            zWhite = np.array([self.ransacW.estimator_.coef_, self.ransacW.estimator_.intercept_])

            #zWhite = np.polyfit(whitePointsArray[:,0],whitePointsArray[:,1],deg=1) # Least-squares fit straight line

            m = np.asscalar(zWhite[0])
            c = np.asscalar(zWhite[1])
            phiWhite = np.arcsin(-m/(np.sqrt(1 + m**2)))
            r_wz_b = np.array([[-c*m/(1+m**2)],[c-(c*m**2/(1+m**2))]]) # Position vector to nearest point on the line (normal to line)
            C_wb = np.array([[np.cos(phiWhite),-np.sin(phiWhite)],[np.sin(phiWhite), np.cos(phiWhite)]])
            r_wz_w = np.matmul(C_wb,r_wz_b) # Distance TO white lane from duckiebot. This should be negative


            if r_wz_w[1] >= 0: # Then white line is on our left. 
                # This can occur in two situations
                # 1) We are in the other lane
                # 2) We are turning a right corner.
                # In either case, it is actually the OTHER white line
                r_dw_w = np.array([[0],[-1.5*laneWidth]])
            else:
                r_dw_w = np.array([[0],[laneWidth]])


            r_zd_w = -r_wz_w - r_dw_w
            distWhite = r_zd_w[1] # Distance TO duckiebot from desired point.

        if nYellow >= 2:  # Need minimum of two points to define a line.
            self.ransacY.fit(np.reshape(yellowPointsArray[:,0],(-1,1)), np.array(np.reshape(yellowPointsArray[:,1],(-1,1))))
            zYellow = np.array([self.ransacY.estimator_.coef_, self.ransacY.estimator_.intercept_])
            #print(zYellow)
            #zWhite = np.polyfit(whitePointsArray[:,0],whitePointsArray[:,1],deg=1) # Least-squares fit straight line

            m = np.asscalar(zYellow[0])
            c = np.asscalar(zYellow[1])
            phiYellow = np.arcsin(-m/(np.sqrt(1 + m**2)))
            r_yz_b = np.array([[-c*m/(1+m**2)],[c-(c*m**2/(1+m**2))]])
            C_yb = np.array([[np.cos(phiYellow),-np.sin(phiYellow)],[np.sin(phiYellow), np.cos(phiYellow)]])
            r_yz_y = np.matmul(C_yb,r_yz_b)
            r_dy_y = np.array([[0],[-laneWidth]])
            r_zd_y = -r_yz_y - r_dy_y
            distYellow = r_zd_y[1] # Distance TO duckiebot from desired point.
       
        # Combination logic
        if nWhite >= 2 and nYellow >= 2:
            #phi = (phiWhite + phiYellow)/2
            phi =(phiYellow)
            d =  (distYellow)
            pa1 = self.line2PointArray(zWhite)
            pa2 = self.line2PointArray(zYellow)
            self.visualizeCurves(pa1,pa2)
        elif nWhite >= 2:
            phi = phiWhite
            d = distWhite + laneWidth/2
            pa1 = self.line2PointArray(zWhite)
            self.visualizeCurves(pa1)
        elif nYellow >= 2:
            phi = phiYellow 
            d = distYellow - laneWidth/2
            pa2 = self.line2PointArray(zYellow)
            self.visualizeCurves(pa2)
        else:
            phi = 0
            d = 0
        
        # ###### PUBLISH LANE POSE #######
        lanePose = LanePose()
        lanePose.header.stamp = segListMsg.header.stamp
        lanePose.d = d
        lanePose.phi = phi
        lanePose.in_lane = True
        lanePose.status = lanePose.NORMAL
        self.pub_lane_pose.publish(lanePose)

        # ####### VISUALIZE ########
    

        # ######### PRINT ##########
        #" phiWhite: %.2f" % phiWhite," phiYellow: %.2f" % phiYellow,
       # print(whitePointsArray)
        #print(whitePointsArray)
        #print("Iter: %d" % self.testCount," nWhite: %.2f" % nWhite, " nYellow: %.2f" % nYellow," phi: %.2f" % phi," distWhite: %.2f" % distWhite," distYellow: %.2f" % distYellow," d %.2f" % d) 
    
    def predict(self,carCmdMsg):
        v = carCmdMsg.v 
        w = carCmdMsg.omega

    # ##########################################################################
    # ######################## VISUALIZATION FUNCTIONS ####################### #
    # ##########################################################################
    def visualizeLine(self,coeffs):
        p1 = np.array([0,coeffs[1]])
        p2 = np.array([1,coeffs[0]+coeffs[0]])  
        pointArray = np.array([p1,p2])
        marker_array = MarkerArray()
        marker_array.markers.append(self.pointArray2Marker(pointArray))
        # rospy.loginfo("[%s] publishing %s marker."%(self.node_name,len(marker_array.markers)))
        self.pub_lane.publish(marker_array)

    def visualizeCurves(self,*pointArrays):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = self.veh_name
        marker.ns = self.veh_name + "/line_seg"
        marker.id = 0
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration.from_sec(5.0)
        marker.type = Marker.LINE_LIST
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.02
        for pointArray in pointArrays:
            arrayShape = np.shape(pointArray)
            for lv1 in range(arrayShape[1]-1):
                seg = Segment()
                seg.points[0].x = pointArray[lv1][0]
                seg.points[0].y = pointArray[lv1][1]
                seg.points[1].x = pointArray[lv1+1][0]
                seg.points[1].y = pointArray[lv1+1][1]
                marker.points.append(seg.points[0])
                marker.points.append(seg.points[1])
                color = ColorRGBA(r=1.0,g=0.0,b=0.0,a=1.0)
                marker.colors.append(color)
                marker.colors.append(color)

        marker_array.markers.append(marker)
        self.pub_lane.publish(marker_array)

    def pointArray2Marker(self,pointArray):
        marker = Marker()
        marker.header.frame_id = self.veh_name
        marker.ns = self.veh_name + "/line_seg"
        marker.id = 0
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration.from_sec(5.0)
        marker.type = Marker.LINE_LIST
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.02
        arrayShape = np.shape(pointArray)
        for lv1 in range(arrayShape[1]-1):
            seg = Segment()
            seg.points[0].x = pointArray[lv1][0]
            seg.points[0].y = pointArray[lv1][1]
            seg.points[1].x = pointArray[lv1+1][0]
            seg.points[1].y = pointArray[lv1+1][1]
            # point_start = seg.points[0]
            # point_end = seg.points[1]
            marker.points.append(seg.points[0])
            marker.points.append(seg.points[1])
            color = ColorRGBA(r=1.0,g=0.0,b=0.0,a=1.0)
            marker.colors.append(color)
            marker.colors.append(color)

        return marker

    def pointArrays2Marker(self,*pointArrays):
        marker = Marker()
        marker.header.frame_id = self.veh_name
        marker.ns = self.veh_name + "/line_seg"
        marker.id = 0
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration.from_sec(5.0)
        marker.type = Marker.LINE_LIST
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.02
        for pointArray in pointArrays:
            arrayShape = np.shape(pointArray)
            print(pointArray)
            for lv1 in range(arrayShape[1]-1):
                seg = Segment()
                seg.points[0].x = pointArray[lv1][0]
                seg.points[0].y = pointArray[lv1][1]
                seg.points[1].x = pointArray[lv1+1][0]
                seg.points[1].y = pointArray[lv1+1][1]
                marker.points.append(seg.points[0])
                marker.points.append(seg.points[1])
                color = ColorRGBA(r=1.0,g=0.0,b=0.0,a=1.0)
                marker.colors.append(color)
                marker.colors.append(color)

        return marker

    def segList2Marker(self,seg_list_msg):
        marker = Marker()
        marker.header.frame_id = self.veh_name
        marker.header.stamp = seg_list_msg.header.stamp
        marker.ns = self.veh_name + "/line_seg"
        marker.id = 0
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration.from_sec(5.0)
        marker.type = Marker.LINE_LIST
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.02
        for seg in seg_list_msg.segments:
            marker.points.append(seg.points[0])
            marker.points.append(seg.points[1])
            color = self.seg_color_dict[seg.color]
            marker.colors.append(color)
            marker.colors.append(color)
        return marker
    def line2PointArray(self,coeffs):
        p1 = np.array([0,coeffs[1]])
        p2 = np.array([1,coeffs[0]+coeffs[0]])  
        pointArray = np.array([p1,p2])
        return pointArray

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