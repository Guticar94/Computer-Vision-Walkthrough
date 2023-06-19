import numpy as np
import cv2
from PIL import Image

from util import getLimits

# Color to capture in BGR (YELLOW)
COLOR = [0, 255, 255]

# Define camera object
webcam = cv2.VideoCapture(0)
# Visualise webcam
while True:
    # Read from camera
    ret, frame = webcam.read()

    # Convert image to HSV colorspace (Hue Saturation Value)
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get color range limits
    lowerLimit, upperLimit = getLimits(COLOR)
    
    # Mask pixels belonging the color we want to detect: We get all black but the color range we want to detect
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    # Send numpy array to Pillow
    mask_ = Image.fromarray(mask)

    # Get the bounding box
    bbox = mask_.getbbox()

    # Plot the bounding box
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Show Image
    cv2.imshow('img', frame)
    # Close condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destroy windows
webcam.release()
cv2.destroyAllWindows()
