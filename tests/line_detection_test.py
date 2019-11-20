import cv2
import numpy as np
from dtProjectHarris import dtProjectHarris

# DONE : 
# - Detect harris an orb features on various images.  Harris on yellow bw mask
#   seems to be working best for yellow points.
# - Get image coordinates of all keypoints. 

# TODO :
# - Nonmax supression of detected corners
# - Figure out white line detection
# - Make new image processing node
# - Package each point into a `segment' message and output to ground projection
#   node.
# - Pipe the output from ground projection into a GP fit.
# - Pipe the outout of gp fit into estimation node

# Select an image to process
# src = cv2.imread('frame0003.jpg')
# src = cv2.imread('frame0446.jpg')
# src = cv2.imread('frame0747.jpg')
src = cv2.imread('frame0141.jpg')

# Select an image processing choice
# choice = "harris_gray"
choice = "harris_only_yellow"
# choice = "harris_only_white"
# choice = "orb"
# choice = "orb_gray"
# choice = "orb_only_yellow"
# choice = "orb_only_white"

################################################

# Crop the image, a little more than line_detector_node (50 vs. 40)
top_cutoff = 50
src_cropped = src[top_cutoff:, :, :]

# Greyscale of cropped
gray_cropped = cv2.cvtColor(src_cropped,cv2.COLOR_BGR2GRAY)

# Try bilateral filter on cropped images (bw and rbg)
# See: http://people.csail.mit.edu/sparis/bf_course/
# Diameter of pixel neighbourhood
d = 7
sigmaColor = 75
sigmaSpace = 75
colour_cropped_smoothed = cv2.bilateralFilter(src_cropped, d, sigmaColor, sigmaSpace)
gray_cropped_smoothed = cv2.bilateralFilter(gray_cropped, d, sigmaColor, sigmaSpace)

# Create a mask for yellow points.  Use same process as _colorFilter(), in
# line_detector2.py
hsv_yellow1 = np.array([25,140,100])
hsv_yellow2 = np.array([45,255,255])
img_hsv = cv2.cvtColor(colour_cropped_smoothed, cv2.COLOR_BGR2HSV)
cropped_yellow_mask = cv2.inRange(img_hsv, hsv_yellow1, hsv_yellow2)

# Create a mask for white points.
hsv_white1 = np.array([0,0,150])
hsv_white2 = np.array([180,60,255])
cropped_white_mask = cv2.inRange(img_hsv, hsv_white1, hsv_white2)

# TRY : Harris corners and corner locations
if choice == "harris_gray" or choice == "harris_only_yellow" or \
    choice == "harris_only_white":
    if choice == "harris_gray":
        dts, corner_xy = dtProjectHarris(gray_cropped_smoothed)
    if choice == "harris_only_yellow":
        dts, corner_xy = dtProjectHarris(cropped_yellow_mask)
    if choice == "harris_only_white":
        dts, corner_xy = dtProjectHarris(cropped_white_mask)

# TRY : Orb features and feature locations
if choice == "orb" or choice == "orb_gray" or choice == "orb_only_yellow" or \
    choice == "orb_only_white":
    orb = cv2.ORB_create(nfeatures=250)
    if choice == "orb":
        kps_orb, desc_orb = orb.detectAndCompute(colour_cropped_smoothed, None)
    if choice == "orbg_ray":
        kps_orb, desc_orb = orb.detectAndCompute(gray_cropped_smoothed, None)
    if choice == "orb_only_yellow":
        kps_orb, desc_orb = orb.detectAndCompute(cropped_yellow_mask, None)
    if choice == "orb_only_white":
        kps_orb, desc_orb = orb.detectAndCompute(cropped_white_mask, None)

    # Display orb features on the cropped, smoothed colour image
    dts = cv2.drawKeypoints(colour_cropped_smoothed, kps_orb, None)
    # Pull out the pixel coordinates of the orb features
    orb_xy = np.zeros((len(kps_orb), 2))
    for lv1 in range(len(kps_orb)):  
        orb_xy[lv1, :] = [kps_orb[lv1].pt[0], kps_orb[lv1].pt[1]]

# Print the image
cv2.imwrite('output.jpg', dts)