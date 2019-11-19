import cv2
import duckietown_utils as dtu

import numpy as np

from .line_detector_interface import Detections, LineDetectorInterface


class LanePointDetector(dtu.Configurable, LineDetectorInterface):
    """ LanePointDetector """

    def __init__(self, configuration):
        # Images to be processed
        self.bgr = np.empty(0)
        self.smoothed = np.empty(0)
        self.hsv = np.empty(0)
        self.gray = np.empty(0)

        param_names = [
            'hsv_white1',
            'hsv_white2',
            'hsv_yellow1',
            'hsv_yellow2',
            'hsv_red1',
            'hsv_red2',
            'hsv_red3',
            'hsv_red4',
            'bilateral_pixel_neighbourhood',
            'bilateral_sigma_color',
            'bilateral_sigma_space',
            'max_corners',
            'yellow_quality_level',
            'yellow_min_distance',
            'yellow_block_size',
            'white_quality_level',
            'white_min_distance',
            'white_block_size',
            'red_quality_level',
            'red_min_distance',
            'red_block_size',
        ]

        dtu.Configurable.__init__(self, param_names, configuration)

    def _smooth(self, bgr):
        # Smooth the bgr image using bilateral filter
        img_smoothed = cv2.bilateralFilter(bgr, self.bilateral_pixel_neighbourhood, \
            self.bilateral_sigma_color, self.bilateral_sigma_space)
        return img_smoothed

    def _getMask(self, color):
        # Get mask by thresholding colours in HSV space
        if color == 'white':
            mask = cv2.inRange(self.hsv, self.hsv_white1, self.hsv_white2)
        elif color == 'yellow':
            mask = cv2.inRange(self.hsv, self.hsv_yellow1, self.hsv_yellow2)
        elif color == 'red':
            mask1 = cv2.inRange(self.hsv, self.hsv_red1, self.hsv_red2)
            mask2 = cv2.inRange(self.hsv, self.hsv_red3, self.hsv_red4)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            raise Exception('Error: Undefined color strings...')
        return mask

    def _getKeypoints(self, mask, color):
        # Set detector parameters depending on line colour
        if color == 'white':
            quality_level = self.white_quality_level
            min_distance = self.white_min_distance
            block_size = self.white_block_size
        elif color == 'yellow':
            quality_level = self.yellow_quality_level
            min_distance = self.yellow_min_distance
            block_size = self.yellow_block_size
        elif color == 'red':
            # TODO : Set these values in dt_project.yaml
            quality_level = self.red_quality_level
            min_distance = self.red_min_distance
            block_size = self.red_block_size
        else:
            raise Exception('Error: Undefined color strings...')
        # Identify Shi-Tomasi keypoints in the greyscale image
        kps = cv2.goodFeaturesToTrack(self.gray, self.max_corners, quality_level, \
            min_distance, mask=mask, blockSize=block_size)
        if kps is None:
            kps = []
        return kps

    def _fakeReturns(self, kps):
        # Fake the data needed by the segment list
        lines = np.zeros((len(kps), 4))
        normals = np.zeros((len(kps), 2))
        centres = kps
        return lines, normals, centres

    def detectLines(self, color):
        with dtu.timeit_clock('_getMask'):
            mask = self._getMask(color)
        with dtu.timeit_clock('_getKeypoints'):
            kps = self._getKeypoints(mask, color)
        with dtu.timeit_clock('_fakeReturns'):
            lines, normals, centers = self._fakeReturns(kps)
        return Detections(lines=lines, normals=normals, area=mask, centers=centers)

    def setImage(self, bgr):
        # Smooth image, convert to hsv, and convert to grayscale. 
        with dtu.timeit_clock('np.copy'):
            self.bgr = np.copy(bgr)
        with dtu.timeit_clock('bilateral smoothing'):
            self.smoothed = self._smooth(bgr)
        with dtu.timeit_clock('cvtColor COLOR_BGR2HSV'):
            self.hsv = cv2.cvtColor(self.smoothed, cv2.COLOR_BGR2HSV)
        with dtu.timeit_clock('_greyscale'):
            self.gray = cv2.cvtColor(self.smoothed, cv2.COLOR_BGR2GRAY)

    def getImage(self):
        return self.bgr
