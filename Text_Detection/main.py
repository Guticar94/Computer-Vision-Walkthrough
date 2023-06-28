import cv2
import easyocr

# Read Image (Get image to read)
path = './Text_Detection/Input/img3.png'
img = cv2.imread(path)

# Instance and tex editor (Use of GPU)
reader = easyocr.Reader(['en'], gpu=True)

# Detect text on image
text_ = reader.readtext(img)

# Threshold confidence value
threshold = 0.3

# For each text bbox
for t in text_:
    try:
        # Save the bounding box, the text and the score
        bbox, text, score = t
        # Threshold filter
        if score > threshold:
            # Add thetext and the bounding boxes to the images
            cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 3)
            cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
    except:
        continue

# Show the image
cv2.imshow('img', img)
cv2.waitKey(0)    



