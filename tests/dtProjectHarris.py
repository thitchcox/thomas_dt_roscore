import cv2
import numpy as np

# Try Harris corners
# TH NOTES
# The following settings do a pretty good job of detecting yellow lane corners in the image. 
# Detector is pretty good at discriminating between corners (lots of hits in lane centre) and 
# lines (almost none for white line, see test image 446).  
# Could combine this with a mask, and use to detect only yellow line segments?
def dtProjectHarris(src_bw_smoothed):

    blockSize = 5
    apertureSize = 7
    k = 0.04
    dst = cv2.cornerHarris(src_bw_smoothed, blockSize, apertureSize, k)

    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dst_harris = cv2.convertScaleAbs(dst_norm)
 
    # TODO : Nonmax supression of corners.  See :
    # https://answers.opencv.org/question/186538/to-find-the-coordinates-of-corners-detected-by-harris-corner-detection/
    
    thresh = 75
    corner_xy = np.argwhere(dst_harris > thresh)

    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i,j]) > thresh:
                cv2.circle(dst_harris, (j,i), 5, (0), 2)

    return dst_harris, corner_xy