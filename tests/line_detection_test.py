import cv2
import numpy as np
from dtProjectHarris import dtProjectHarris
import time

from skimage.morphology import skeletonize
from skimage.util import invert
# from skimage import data

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
my_dir = './training_data/long_run/'
src = cv2.imread(my_dir + 'frame0110.jpg')
# src = cv2.imread('frame0446.jpg')
# src = cv2.imread('frame0747.jpg')
# src = cv2.imread('frame0141.jpg')

# Select an image processing choice
# choice = "harris_gray"
# choice = "harris_only_yellow"
# choice = "harris_only_white"
# choice = "orb"
# choice = "orb_gray"
# choice = "orb_only_yellow"
# choice = "orb_only_white"
# choice = "gf2t_yellow" 
# choice = "gf2t_white" 
choice = "white_line_detection"

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

time_start = time.time()

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

# TRY : Good features to track
if choice == "gf2t_yellow" or choice == "gf2t_white":
    # Detecting yellow points
    if choice == "gf2t_yellow":
        max_corners = 50
        quality_level = 0.01
        min_distance = 5
        block_size = 7
        corner_xy = cv2.goodFeaturesToTrack(gray_cropped_smoothed, max_corners, quality_level, min_distance, mask=cropped_yellow_mask, \
            blockSize=block_size, useHarrisDetector=0, k=0.04)

    # Detecting white points
    if choice == "gf2t_white":
        max_corners = 50
        quality_level = 0.001
        min_distance = 10
        block_size = 15
        corner_xy = cv2.goodFeaturesToTrack(gray_cropped_smoothed, max_corners, quality_level, min_distance, mask=cropped_white_mask, \
            blockSize=block_size, useHarrisDetector=0, k=0.04)
    
    # Show corners on source image
    dts = src_cropped.copy()
    corner_xy = np.int0(corner_xy)
    for lv1 in corner_xy:
        x,y = lv1.ravel()
        cv2.circle(dts, (x,y), 3, 255, -1)

if choice == "white_line_detection":
    # First dilate to connect segments
    dilatation_type = 0
    dilatation_size = 3
    element = cv2.getStructuringElement(dilatation_type, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
    dilatation_dst = cv2.dilate(cropped_white_mask, element)

    # Get connected components
    n_comps, output, stats, _ = cv2.connectedComponentsWithStats(dilatation_dst, connectivity=8)
    # Remove background
    sizes = stats[1:, -1]
    n_comps = n_comps - 1
    min_size = 100
    dts = np.zeros((output.shape), dtype=np.uint8)
    print("type dts before:", dts.dtype)

    for i in range(0, n_comps):
        if sizes[i] >= min_size:
            dts[output == i + 1] = 255

    # Skeletonization
    # skel = 255 * skeletonize(dts)

    # Detect white lines
    #Create default parametrization LSD
    lsd = cv2.createLineSegmentDetector(0)
    print("type dts:", dts.dtype)
    print("type mask: ", cropped_white_mask.dtype)
    # print(skel)
    # #Detect lines in the image
    # lines = lsd.detect(cropped_white_mask)[0] #Position 0 of the returned tuple are the detected lines
    edges = cv2.Canny(dts, 80, 200, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 2, np.empty(1), 3, 1)


    # #Draw detected lines in the image
    drawn_img = lsd.drawSegments(0*cropped_white_mask, lines)

# Printing and debugging.
time_end = time.time()
print("Execution time is", time_end - time_start)
# print("List of detected corners", corner_xy)
# Write the image to file

cv2.imwrite('output_lines.jpg', drawn_img)
# cv2.imwrite('output_skel.jpg', skel)