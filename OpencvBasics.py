import cv2
import os
import numpy as np

# Source: https://www.youtube.com/watch?v=eDIj5LuIL4A

# image path
imagePath = os.path.join('.', 'images', 'me.jpg')

# Image numpy array
img = cv2.imread(imagePath)

#  ==========================================================================
# # # Image resize
def resize(img):
    # Resize the image
    imgResize = cv2.resize(img, (round(1081/2), round(1080/2)))
    
    # Visualize the images
    cv2.imshow('img_', img)
    cv2.imshow('imgResize', imgResize)
    cv2.waitKey(0)
    print(img_resize.shape)

#  ==========================================================================
# # Image cropping
def crop(img, imgResize):
    imgCrop = img[50:700, 300:780]

    # Visualize the images
    cv2.imshow('img', img)
    cv2.imshow('imgResize', imgResize)
    cv2.imshow('imgCrop', imgCrop)
    cv2.waitKey(0)
    print(imgCrop.shape)

#  ==========================================================================
# image color
def coloring(img):
    imgColorRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgColorGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgColorHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Visualize the images
    cv2.imshow('Image', img)
    cv2.imshow('imgColorRGB', imgColorRGB)
    cv2.imshow('imgColorGray', imgColorGray)
    cv2.imshow('imgColorHSV', imgColorHSV)
    cv2.waitKey(0)

#  ==========================================================================
# Image threshhold
def threshold(img):
    imgColorGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    RET, imgTresh = cv2.threshold(imgColorGray, 120, 255, cv2.THRESH_BINARY)

    # Visualize the images
    cv2.imshow('img', img)
    cv2.imshow('imgTresh', imgTresh)
    cv2.waitKey(0)

#  ==========================================================================
def edgeDetection(img):
    imgEdge = cv2.Canny(img, 100, 50)

    # Optional dilate: Dilates the borders of the image, we make the edges thicker
    imgDilate = cv2.dilate(imgEdge, np.ones((3, 3), dtype = np.int8))

    # Optional erode: Erodes the borders of the image, we make the edges thinner
    imgErode = cv2.erode(imgDilate, np.ones((3, 3), dtype = np.int8))

    # Visualize the images
    cv2.imshow('img', img)
    cv2.imshow('imgDilate', imgDilate)
    cv2.imshow('imgErode', imgErode)
    cv2.imshow('imgEdge', imgEdge)
    cv2.waitKey(0)

#  ==========================================================================
# Image drawing
def imageDrawing(img):
    # Draw a line
    cv2.line(img, (100, 150), (300, 450), (0, 255, 0), 3)

    # Draw a rectangle
    cv2.rectangle(img, (100, 100), (1000,600), (0, 0, 255), 5)

    # Draw a circle
    cv2.circle(img, (550, 350), 300, (255, 0, 0), 10)

    # Draw text
    cv2.putText(img, 'This is a test text', (100, 600), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 3)

     # Visualize the images
    cv2.imshow('img', img)
    cv2.waitKey(0)

#  ==========================================================================
# image contours
def imageContours(img):
    # Transform the image to gray scale
    imgColorGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Get the image threshold: Inverse because we want to encapsulate the contours
    RET, imgTresh = cv2.threshold(imgColorGray, 120, 255, cv2.THRESH_BINARY_INV)
    # Get contours
    contours, hierarchy = cv2.findContours(imgTresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Print contours
    for cnt in contours:
        # If the contour areas are greater than 200
        if cv2.contourArea(cnt) > 200:
            # Draw the contours in the original image
            # cv2.drawContours(img, cnt, -1, (0, 0, 255), 3)
            
            # Get contours bounding boxes
            x1, y1, w, h = cv2.boundingRect(cnt)
            # Ad the bounding boxes rectangles
            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

    # Visualize the images
    cv2.imshow('img', img)
    cv2.waitKey(0)


imageContours(img)