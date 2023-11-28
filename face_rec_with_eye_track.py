import cv2
import numpy as np
from gaze_tracking import GazeTracking
import os 

#facial recognition
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face_rec/trainer/trainer.yml')
cascadePath = "face_rec/cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Matheus'] 
# Initialize and start realtime video capture
webcam = cv2.VideoCapture(0)
# webcam.set(3, 640) # set video widht
# webcam.set(4, 480) # set video height
# # Define min window size to be recognized as a face
minW = 0.1*webcam.get(3)
minH = 0.1*webcam.get(4)


#gaze tracking
gaze = GazeTracking()

WIN = 'Example'

cv2.namedWindow(WIN)

while cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE):
    # We get a new frame from the webcam
    _, frame = webcam.read()
    og_frame = frame.copy()


    #facial recognition
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
    )
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        # If confidence is less then 100 ==> "0" : perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(frame, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
    

    #gaze Tracking
    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"
    elif gaze.is_up():
        text = "Looking up"
    elif gaze.is_down():
        text = "Looking down"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    #cv2.imshow("Demo", frame)

    #creating the double frame
    height, width, num_channels = frame.shape
    image = np.empty((height, 2 * width, num_channels), frame.dtype)
    image[:, :width] = og_frame
    image[:, width:] = frame
    
    if cv2.waitKey(1) == 32:
        image[:, :width] = frame


    if cv2.waitKey(1) == 27:
        break

    cv2.imshow(WIN, image)



webcam.release()
cv2.destroyAllWindows()