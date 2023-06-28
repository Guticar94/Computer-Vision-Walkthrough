import cv2
import matplotlib.pyplot as plt
import mediapipe as mp

# load images
img = cv2.imread('./Image_replacement/Input/img.png')
img_bur = cv2.imread('./Image_replacement/Input/burg.png')

img_bur = img_bur[290:-40, 420:810]
for i in range(400,800,40):
    for j in range(200,500,40):
        imgEdge = cv2.Canny(img_bur, i, j)
        plt.title(str(i)+str(j))
        plt.imshow(cv2.cvtColor(imgEdge, cv2.COLOR_BGR2RGB))
        plt.show()





