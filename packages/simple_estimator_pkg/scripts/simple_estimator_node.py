#!/usr/bin/env python
import math
import rospy
# Import the different message structures that we will use.
from duckietown_msgs.msg import Segment, SegmentList, Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, ColorRGBA #Imports msg
import time
import numpy as np
import scipy as sp
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
        self.sub_baseline_pose = rospy.Subscriber("~baseline_pose",LanePose, self.updateAverage, queue_size=1)
        self.sub_velocity = rospy.Subscriber("~car_cmd", Twist2DStamped, self.predict)
        #self.sub_fsm_mode = rospy.Subscriber("~fsm_mode", FSMState, self.cbMode, queue_size=1)

        # safe shutdown
        rospy.on_shutdown(self.custom_shutdown)
        # ##################################
        # ######## WORKING VARIABLES #######
        # ##################################
        # RANSAC variables
        self.ransacW = linear_model.RANSACRegressor()
        self.ransacW.residual_threshold = 0.1
        self.ransacY = linear_model.RANSACRegressor()
        self.ransacY.residual_threshold = 0.15

        # Inclusion zone for segments
        self.x_min = 0
        self.x_max = 0.3
        self.y_min = -0.4
        self.y_max = 0.4

        # NLS parameters
        self.weightYellow = 5
        self.laneWidth = 0.33
        self.zTwoLane = np.array([0,-self.laneWidth/2])

        # Kalman Filter parameters
        self.previousTime = float(rospy.Time.now().to_sec())
        self.carCmdPrev = Twist2DStamped()
        self.d_k = 0 # Initial
        self.phi_k = 0 
        self.x_k = np.array([[self.d_k],[self.phi_k]])
        self.A_k = np.array([[1,0],[0,1]])
        self.B_k = np.array([[0],[0]])
        self.C_k = np.identity(2)

        # Baseline pose
        self.d_baseline = 0.0
        self.phi_baseline = 0.0

        self.P_k = 5*np.identity(2)
        self.Q = np.array([[1,0],[0,1]])
        self.R = np.identity(2)

        self.testCount = 0

        # ##################################

    # ##########################################################################
    # ######################## MAIN ESTIMATOR FUNCTION ####################### #
    # ##########################################################################
    def updateEstimate(self, segListMsg):

        self.testCount = self.testCount + 1 # Just some counter, gives an idea of the speed.

        # ######## EXTRACT ######### data from segments back into separated arrays.
        whitePointsArray, yellowPointsArray = self.segList2Array(segListMsg)
        nWhite = whitePointsArray.shape[0]
        nYellow = yellowPointsArray.shape[0]
        

        # ###### REMOVE WHITE POINTS OF LEFT LANE ######
        if nYellow >= 2: # Need minimum of two points to define a line.
            self.ransacY.fit(np.reshape(yellowPointsArray[:,0],(-1,1)), np.array(np.reshape(yellowPointsArray[:,1],(-1,1))))
            zYellowOnly = np.array([self.ransacY.estimator_.coef_, self.ransacY.estimator_.intercept_])
            whitePointsArray = self.removeLeftLane(zYellowOnly, whitePointsArray)
        nWhite = whitePointsArray.shape[0]
        nYellow = yellowPointsArray.shape[0]

        # ###### FIT TWO LANES ########
        if nWhite >= 2 and nYellow >= 2: 
            z = self.fitTwoLanes(whitePointsArray,yellowPointsArray)
            #self.zTwoLane = z
            m = z[0]
            c = z[1]
            zWhite = np.array([m,c])
            zYellow = np.array([m,c + self.laneWidth*np.sqrt(m**2 + 1)])
        elif nYellow >= 2:
            zWhite = zYellowOnly
        elif nWhite >= 2:
            self.ransacW.fit(np.reshape(whitePointsArray[:,0],(-1,1)),np.reshape(whitePointsArray[:,1],(-1,1)))
            zWhite = np.array([self.ransacW.estimator_.coef_, self.ransacW.estimator_.intercept_])
            

        if nWhite >=2 or nYellow >= 2:
            # ######### Calculate d and phi from geometry ##########
            m = np.asscalar(zWhite[0])
            c = np.asscalar(zWhite[1])
            phiWhite = np.arcsin(-m/(np.sqrt(1 + m**2)))
            r_wz_b = np.array([[-c*m/(1+m**2)],[c-(c*m**2/(1+m**2))]]) # Position vector to nearest point on the line (normal to line)
            C_wb = np.array([[np.cos(phiWhite),-np.sin(phiWhite)],[np.sin(phiWhite), np.cos(phiWhite)]])
            r_wz_w = np.matmul(C_wb,r_wz_b) # Distance TO white lane from duckiebot. This should be negative
            if nWhite < 2 and nYellow >= 2: # Then the line we have is actually yellow
                r_dw_w = np.array([[0],[-self.laneWidth/2]])
            else:
                if r_wz_w[1] >= 0: # Then white line is on our left. 
                    # This can occur in two situations
                    # 1) We are in the other lane
                    # 2) We are turning a right corner.
                    # In either case, it is actually the OTHER white line
                    r_dw_w = np.array([[0],[-1.5*self.laneWidth]])
                else:
                    r_dw_w = np.array([[0],[self.laneWidth/2]])
            r_zd_w = -r_wz_w - r_dw_w
            distWhite = r_zd_w[1] # Distance TO duckiebot from desired point.
            
            phi = phiWhite
            d = distWhite
        else:
            d = 0
            phi = 0

        # ######### VISUALIZE ###########
        if nWhite >= 2 and nYellow >= 2:
            pa1 = self.line2PointArray(zWhite, 0,1)
            pa2 = self.line2PointArray(zYellow, 0,1)
            self.visualizeCurves(pa1,pa2)
        elif nWhite >= 2 or nYellow >= 2:
            pa1 = self.line2PointArray(zWhite, 0,1)
            self.visualizeCurves(pa1)
        

        # ###### PUBLISH LANE POSE #######
        lanePose = LanePose()
        lanePose.header.stamp = segListMsg.header.stamp

        # Complimentary filter lol
        lanePose.d = 0.8*float(d) + 0.2*float(self.d_baseline)
        lanePose.phi = 0.8*float(phi) + 0.2*float(self.phi_baseline)
        lanePose.in_lane = True
        lanePose.status = lanePose.NORMAL

        self.correct(lanePose) # Kalman Filter Modifications
        lanePose.d = np.asscalar(self.x_k[0][0])
        lanePose.phi = np.asscalar(self.x_k[1][0])
        self.pub_lane_pose.publish(lanePose)

        # ######### PRINT ##########
        #" nWhite: %.2f" % nWhite, " nYellow: %.2f" % nYellow,
        #" phiWhite: %.2f" % phiWhite," phiYellow: %.2f" % phiYellow,
        #" distWhite: %.2f" % distWhite," distYellow: %.2f" % distYellow,
        #print(whitePointsArray)
        #print(whitePointsArray)
        #print("Iter: %d" % self.testCount," phi: %.2f" % phi," d %.2f" % d) 
    def updateAverage(self,lanePoseMsg):
        self.d_baseline = lanePoseMsg.d 
        self.phi_baseline = lanePoseMsg.phi
        #print(self.phi_baseline)
    def predict(self,carCmdMsg):
        v = self.carCmdPrev.v 
        w = self.carCmdPrev.omega
        
        dt = float(rospy.Time.now().to_sec()) - self.previousTime # WARNING: doesnt line up with simulator

        # Predict d, phi, and pack into state
        self.d_k = self.d_k + dt*v*np.sin(self.phi_k)
        self.phi_k = self.phi_k + dt*w
        self.phi_k = (self.phi_k + np.pi) % (2 * np.pi) - np.pi
        self.x_k = np.array([[self.d_k],[self.phi_k]])
        #print('PREDICTING')
        #print(dt*v*np.sin(self.phi_k))
        #print("d: %.2f" % self.d_k," phi: %.2f" % self.phi_k)
        # Predict Covariance
        self.A_k = np.array([[1,dt*v],[0,1]])
        self.B_k = np.array([[0],[dt]])
        self.P_k = np.matmul(self.A_k,np.matmul(self.P_k,self.A_k.T)) + self.Q

        # Save for next iteration
        self.previousTime = float(rospy.Time.now().to_sec())
        self.carCmdPrev = carCmdMsg
    
    def correct(self,lanePoseMsg):
        d = lanePoseMsg.d 
        phi = lanePoseMsg.phi 
        # Measurement vector
        y_k = np.array([[d],[phi]])

        # Kalman Gain
        V_k = np.matmul(self.C_k, np.matmul(self.P_k,self.C_k.T)) + self.R
        K_k = np.matmul(self.P_k,np.matmul(self.C_k.T,np.linalg.inv(V_k)))

        # Correct State
        self.x_k = self.x_k + np.matmul(K_k,y_k - self.x_k)
        self.d_k = self.x_k[0][0]
        self.phi_k = self.x_k[1][0]

        # Correct Covariance
        self.P_k = np.matmul(np.identity(2) - np.matmul(K_k,self.C_k),self.P_k)
        self.P_k = 0.5*(self.P_k + self.P_k.T)
        # print("d: %.2f" % self.d_k," phi: %.2f" % self.phi_k)
        # print('CORRECTING')

    def removeLeftLane(self,zYellowOnly, whitePointsArray):
        tmp = []
        for lv1 in range(whitePointsArray.shape[0]):
            m = zYellowOnly[0]
            c = zYellowOnly[1]
            x = whitePointsArray[lv1][0]
            # Check if y value is larger than yellow line fit.
            if whitePointsArray[lv1][1] < m*x + c:
                tmp.append(whitePointsArray[lv1,:])
        return np.array(tmp)



    # ##########################################################################
    # ######################## VARIOUS OTHER FUNCTIONS ####################### #
    # ##########################################################################

    def segList2Array(self,segListMsg):
        # Unpacks seglist message into two arrays of yellow and white points. (x,y coordinates)
        # Also only adds to array if they are within the limits.
        whitePointsList = []
        yellowPointsList = []

        for segment in segListMsg.segments:
            if  segment.color == segment.WHITE\
                and segment.points[0].x < self.x_max and segment.points[0].x > self.x_min\
                and segment.points[0].y < self.y_max and segment.points[0].y > self.y_min:
                dx = segment.points[0].x - segment.points[1].x
                dy = segment.points[0].y - segment.points[1].y 
                L = np.sqrt(dx ** 2 + dy ** 2)
                m = dy/dx
                whitePointsList.append(np.array([segment.points[0].x,segment.points[0].y,L,m]))
                #whitePointsList.append(np.array([segment.points[1].x,segment.points[1].y]))
            if segment.color == segment.YELLOW\
                and segment.points[0].x < self.x_max and segment.points[0].x > self.x_min\
                and segment.points[0].y < self.y_max and segment.points[0].y > self.y_min:
                dx = segment.points[0].x - segment.points[1].x
                dy = segment.points[0].y - segment.points[1].y 
                L = np.sqrt(dx ** 2 + dy ** 2)
                m = dy/dx
                yellowPointsList.append(np.array([segment.points[0].x,segment.points[0].y,L,m]))
        whitePointsArray = np.array(whitePointsList)
        yellowPointsArray = np.array(yellowPointsList)

        return whitePointsArray, yellowPointsArray

    def fitTwoLanes(self, wData, yData):
        z0 = self.zTwoLane
        z = z0
        if yData.shape[0] > 0 or wData.shape[0]>0:
            #, lambda x: self.jacTwoLanes(x,wData,yData)
            res = sp.optimize.least_squares(lambda x: self.errorTwoLanes(x,wData,yData), z0)
            z = res.x
        else:
            z = np.array([0,-self.laneWidth/2])

        return z
        

    def errorTwoLanes(self, coeffs, wData, yData):
        m = coeffs[0]
        c = coeffs[1]
        L = self.laneWidth

        if wData.shape[0] > 0:
            xWhite = wData[:,0]
            eWhite = wData[:,1] - (m*xWhite + c)

        if yData.shape[0] > 0:    
            xYellow = yData[:,0]
            eYellow = self.weightYellow*(yData[:,1] - (m*xYellow + c + L*np.sqrt(m ** 2 + 1)) )

        if yData.shape[0] > 0 and wData.shape[0]>0:
            e = np.vstack((np.reshape(eWhite,(-1,1)), np.reshape(eYellow,(-1,1))))
        elif yData.shape[0]>0:
            e = np.reshape(eYellow,(-1,1))
        elif wData.shape[0]>0:
            e = np.reshape(eWhite,(-1,1))
        

        return np.reshape(e,-1)

    def jacTwoLanes(self, coeffs, wData, yData):
        m = coeffs[0]
        c = coeffs[1]
        L = self.laneWidth

        if wData.shape[0] > 0:
            xWhite = wData[:,0]
            jacWm = np.reshape(-xWhite,(-1,1))
            jacWc = np.reshape(-np.ones(xWhite.shape),(-1,1))
            jacW = np.hstack((jacWm, jacWc))

        if yData.shape[0] > 0:    
            xYellow = yData[:,0]
            jacYm = self.weightYellow*np.reshape(-xYellow + L*m/(np.sqrt(m**2 + 1)),(-1,1))
            jacYc = self.weightYellow*np.reshape(-np.ones(xYellow.shape),(-1,1))
            jacY = np.hstack((jacYm, jacYc))

        if yData.shape[0] > 0 and wData.shape[0]>0:
            jac = np.vstack((jacW, jacY))
        elif yData.shape[0]>0:
            jac = jacY
        elif wData.shape[0]>0:
            jac = jacW

        return jac























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
        count = 1
        for pointArray in pointArrays:
            for lv1 in range(len(pointArray)-1):
                seg = Segment()
                seg.points[0].x = pointArray[lv1][0]
                seg.points[0].y = pointArray[lv1][1]
                seg.points[1].x = pointArray[lv1+1][0]
                seg.points[1].y = pointArray[lv1+1][1]
                marker.points.append(seg.points[0])
                marker.points.append(seg.points[1])
                if count == 1:
                    color = ColorRGBA(r=1.0,g=1.0,b=1.0,a=1.0)
                elif count == 2:
                    color = ColorRGBA(r=0.0,g=1.0,b=1.0,a=1.0)
                else:
                    color = ColorRGBA(r=1,g=0, b=0,a=1.0)
                marker.colors.append(color)
                marker.colors.append(color)
                marker_array.markers.append(marker)
            count = count + 1

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

    def line2PointArray(self,coeffs, x1, x2):
        p1 = np.array([x1,coeffs[0]*x1 + coeffs[1]])
        p2 = np.array([x2,coeffs[0]*x2 + coeffs[1]])  
        pointArray = np.array([p1,p2])
        return pointArray

    def poly2PointArray(self, coeffs,x1,x2, N = 20):
        x = np.linspace(x1,x2,N)
        y = np.zeros(x.shape)
        n = coeffs.shape[0]
        for lv1 in range(n):
            y = y + coeffs[n - lv1 - 1]* ( x ** lv1)
        return np.vstack((x,y)).T

    def ransac_polyfit(self, x, y, deg=3, n=3, k=20, t=0.05, d=3, f=0.1): 
        # n minimum number of data points required to fit the model
        # k maximum number of iterations allowed in the algorithm
        # t threshold value to determine when a data point fits a model
        # d number of close data points required to assert that a model fits well to data
        # f fraction of close data points required       
        besterr = 999999
        bestfit = None
        for kk in range(k):
            maybeinliers = np.random.randint(len(x), size=n)
            maybemodel = np.polyfit(x[maybeinliers], y[maybeinliers], deg)
            alsoinliers = np.abs(np.polyval(maybemodel, x)-y) < t
            if sum(alsoinliers) > d and sum(alsoinliers) > len(x)*f:
                bettermodel = np.polyfit(x[alsoinliers], y[alsoinliers], deg)
                thiserr = np.sum(np.abs(np.polyval(bettermodel, x[alsoinliers])-y[alsoinliers]))
                if thiserr < besterr:
                    bestfit = bettermodel
                    besterr = thiserr
        return bestfit

    # ############################# CUSTOM SHUTDOWN ############################
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





        # ####### Gauss Newton optimization #######
        #dz = 10
        #lv1 = 0
        # while np.linalg.norm(dz) > 0.001 and lv1 < 10:
        #     print("GN ITERATING")
        #     e = self.errorTwoLanes(z,wData,yData)
        #     jac = self.jacTwoLanes(z,wData,yData)

        #     hessZ = np.matmul(jac.T, jac)
        #     jacZ = np.matmul(-jac.T,e)
        #     dz = np.linalg.solve(hessZ,jacZ)

        #     # If dz is nan, for whatever reason
        #     if np.isnan(dz).any():
        #         # Exits code
        #         dz = np.array([[0],[0]])

        #     z = z + dz.T
        #     z = np.reshape(z,-1)

        #     # Cap possible values
        #     m_lim = 10
        #     c_lim = 50
        #     z[0] = max(min(z[0],m_lim),-m_lim)
        #     z[1] = max(min(z[1],c_lim),-c_lim)

        #     lv1 = lv1 + 1






    # # ######### FIT ########### straight lines to data
    #     wShape = np.shape(whitePointsArray)
    #     yShape = np.shape(yellowPointsArray)
    #     nWhite = wShape[0]
    #     nYellow = yShape[0]
    #     phiWhite = 0
    #     phiYellow = 0
    #     distWhite = 0
    #     distYellow = 0
    #     if nWhite >= 2: # Need minimum of two points to define a line.
    #         # RANSAC Straight line fit
    #         self.ransacW.fit(np.reshape(whitePointsArray[:,0],(-1,1)),np.reshape(whitePointsArray[:,1],(-1,1)))
    #         #zWhite = np.array([self.ransacW.estimator_.coef_, self.ransacW.estimator_.intercept_])

    #         # 3rd degree polynomial
    #         #polyW = np.polyfit(whitePointsArray[:,0],whitePointsArray[:,1],deg=3) # Least-squares fit 3rd degree polynomial

    #         # Calculate d and phi from geometry
    #         m = np.asscalar(zWhite[0])
    #         c = np.asscalar(zWhite[1])
    #         phiWhite = np.arcsin(-m/(np.sqrt(1 + m**2)))
    #         r_wz_b = np.array([[-c*m/(1+m**2)],[c-(c*m**2/(1+m**2))]]) # Position vector to nearest point on the line (normal to line)
    #         C_wb = np.array([[np.cos(phiWhite),-np.sin(phiWhite)],[np.sin(phiWhite), np.cos(phiWhite)]])
    #         r_wz_w = np.matmul(C_wb,r_wz_b) # Distance TO white lane from duckiebot. This should be negative
    #         if r_wz_w[1] >= 0: # Then white line is on our left. 
    #             # This can occur in two situations
    #             # 1) We are in the other lane
    #             # 2) We are turning a right corner.
    #             # In either case, it is actually the OTHER white line
    #             r_dw_w = np.array([[0],[-1.5*self.laneWidth]])
    #         else:
    #             r_dw_w = np.array([[0],[self.laneWidth/2]])
    #         r_zd_w = -r_wz_w - r_dw_w
    #         distWhite = r_zd_w[1] # Distance TO duckiebot from desired point.

    #     if nYellow >= 2:  # Need minimum of two points to define a line.
    #         # RANSAC straight line fit
    #         self.ransacY.fit(np.reshape(yellowPointsArray[:,0],(-1,1)), np.array(np.reshape(yellowPointsArray[:,1],(-1,1))))
    #         #zYellow = np.array([self.ransacY.estimator_.coef_, self.ransacY.estimator_.intercept_])

    #         # 3rd degree polynomial fit
    #         #polyY = np.polyfit(yellowPointsArray[:,0],yellowPointsArray[:,1],deg=3) # Least-squares fit polynomial

    #         # Calculate d and phi from geometry
    #         m = np.asscalar(zYellow[0])
    #         c = np.asscalar(zYellow[1])
    #         phiYellow = np.arcsin(-m/(np.sqrt(1 + m**2)))
    #         r_yz_b = np.array([[-c*m/(1+m**2)],[c-(c*m**2/(1+m**2))]])
    #         C_yb = np.array([[np.cos(phiYellow),-np.sin(phiYellow)],[np.sin(phiYellow), np.cos(phiYellow)]])
    #         r_yz_y = np.matmul(C_yb,r_yz_b)
    #         r_dy_y = np.array([[0],[-self.laneWidth/2]])
    #         r_zd_y = -r_yz_y - r_dy_y
    #         distYellow = r_zd_y[1] # Distance TO duckiebot from desired point.
       
    #     # ######### COMBINE ###########
    #     if nWhite >= 2 or nYellow >= 2:
    #         phi = (phiYellow + phiWhite)/2 # Just always pick yellow
    #         d =  (distYellow + distWhite)/2
    #         pa1 = self.line2PointArray(zWhite, 0,1)
    #         pa2 = self.line2PointArray(zYellow, 0,1)
    #         #pa3 = self.poly2PointArray(polyW, min(whitePointsArray[:,0]),max(whitePointsArray[:,0]))
    #         #pa4 = self.poly2PointArray(polyY, min(yellowPointsArray[:,0]),max(yellowPointsArray[:,0]))
    #         self.visualizeCurves(pa1,pa2)
    #     elif nWhite >= 2:
    #         phi = phiWhite
    #         d = distWhite + self.laneWidth/2
    #         #pa1 = self.line2PointArray(zWhite, min(whitePointsArray[:,0]),max(whitePointsArray[:,0]))
    #         #pa3 = self.poly2PointArray(polyW, min(whitePointsArray[:,0]),max(whitePointsArray[:,0]))
    #         #self.visualizeCurves(pa1)
    #     elif nYellow >= 2:
    #         phi = phiYellow 
    #         d = distYellow - self.laneWidth/2
    #         #pa2 = self.line2PointArray(zYellow, min(yellowPointsArray[:,0]),max(yellowPointsArray[:,0]))
    #         #pa4 = self.poly2PointArray(polyY, min(yellowPointsArray[:,0]),max(yellowPointsArray[:,0]))
    #         #self.visualizeCurves(pa2)
    #     else:
    #         phi = 0
    #         d = 0
        