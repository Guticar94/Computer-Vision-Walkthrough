import cv2
import mediapipe as mp
import os
import argparse

# Create argument parser object
args = argparse.ArgumentParser()
# Add mode argument and default value in Image
args.add_argument('--mode', default='Webcam')
# Add file path to read and default input
args.add_argument('--filePath', default='./Face_Anonimizer/input/video.mp4')
# Inicialize arguments
args = args.parse_args()

# Define output folder
output_dir = './Face_Anonimizer/output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to detect and blur faces
def faceBlur(img, face_detection):
    # Get Image shape values
    H, W, _ = img.shape
    # Transform color space to RGB for mediapipe use
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Process the image with mediapipe to get faces BBOX
    out = face_detection.process(imgRGB)
    # If any face is found
    if out.detections is not None:
        # Get parameters
        for detection in out.detections:
            # Get bbox locations
            locationData = detection.location_data
            # Save bbox
            bbox = locationData.relative_bounding_box
            # Store coordinates vatiables
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            # Scale coordinates: In mediapipe they are rescaled
            x1 = int(x1*W)
            y1 = int(y1*H)
            w = int(w*W)
            h = int(h*H)
            # Blur the image
            img[y1:y1+h, x1:x1+w] = cv2.blur(img[y1:y1+h, x1:x1+w],(100, 100))
    return img

# Detect Faces: mediapipe face detection
mpDetect = mp.solutions.face_detection

# Open the face detector and start to detect
with mpDetect.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    # If the user inpuetes an Image
    if args.mode in ['Image']:
        # Read Image
        img = cv2.imread(args.filePath)
        # Apply image blurring function
        img = faceBlur(img, face_detection)
        # Save Image
        cv2.imwrite(os.path.join(output_dir, 'output.png'), img)
    
    # If the user inputes a video
    elif args.mode in ['Video']:
        # Capture Video
        video = cv2.VideoCapture(args.filePath)
        ret, frame = video.read()
        # Object to save the video
        outputVideo = cv2.VideoWriter(
            os.path.join(output_dir, 'output.mp4'),     # Specify output directory
            cv2.VideoWriter_fourcc(*'MP4V'),            # Specify writer method
            60,                                         # Change for the video frames FPS
            (frame.shape[1], frame.shape[0]))           # Video shape
        # Loop to iterate through the video
        while ret:
            # Apply image blurring function
            frame = faceBlur(frame, face_detection)
            # Write the frime to the video object
            outputVideo.write(frame)
            # Read next frame
            ret, frame = video.read()
        # Release memory of two created objects
        video.release()
        outputVideo.release()

    # If the user inputes a video
    elif args.mode in ['Webcam']:
        # Instanciate camera
        cam = cv2.VideoCapture(0)
        # Capture Camera
        ret, frame = cam.read()
        # Loop to iterate through the camera video frames
        while ret:
            # Apply image blurring function
            frame = faceBlur(frame, face_detection)
            # Show webcam frames
            cv2.imshow('Webcam', frame)
            # Close condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # Read next frame
            ret, frame = cam.read()
        # Release memory and destroy windows
        cam.release()
        cv2.destroyAllWindows()