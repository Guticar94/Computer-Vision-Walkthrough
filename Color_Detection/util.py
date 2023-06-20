import cv2
import numpy as np

def getLimits(color):
    # Get the color in BGR
    c = np.uint8([[color]])
    # Conver to HSV
    hsvColor = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    # Get lower limits
    lowerLimit = np.array((hsvColor[0][0][0] - 10, 100, 100), dtype = np.uint8)
    # Get Higger limits
    upperLimit = np.array((hsvColor[0][0][0] + 10, 255, 255), dtype = np.uint8)

    return lowerLimit, upperLimit
