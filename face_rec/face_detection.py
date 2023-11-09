import numpy as np
import cv2
import json

faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

users = []
with open('users.json') as json_file:
    users = json.load(json_file)
face_id = input('\nEnter user id and press <return> ==>  ')
if face_id in users:
    print(f"\nUser {face_id} already exists")
    exit()
print("\n[INFO] Initializing face capture. Look the camera and wait ...")
count = 0 # Initialize individual sampling face count

while(True):
    ret, img = cap.read()
    face_region = img[80:400, 160:480]
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(face_region,(x,y),(x+w,y+h),(255,0,0),2)
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y+h,x:x+w])
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = face_region[y:y+h, x:x+w]
        roi_left_eye = roi_gray[int(h*0.2):int(h*0.6), 0:int(w/2)]
        roi_right_eye = roi_gray[int(h*0.2):int(h*0.6), int(w/2):w]
        
        left_eye = eyeCascade.detectMultiScale(
            roi_left_eye,
            scaleFactor= 1.5,
            minNeighbors=10,
            minSize=(5, 5),
        )
        
        if len(left_eye) > 0:
            biggest_left = left_eye[0]
            biggest_area = (biggest_left[2]-biggest_left[0])*(biggest_left[3]-biggest_left[1])
            if len(left_eye) > 1:
                for (ex, ey, ew, eh) in left_eye:
                    if (ew-ex)*(eh-ey) > biggest_area:
                        biggest_area = (ew-ex)*(eh-ey)
                        biggest_left = (ex, ey, ew, eh)
            (ex, ey, ew, eh) = biggest_left
            # cv2.rectangle(roi_color, (ex, int(h*0.2) + ey), (ex + ew, int(h*0.2) + ey + eh), (0, 255, 0), 2)
        
        right_eye = eyeCascade.detectMultiScale(
            roi_right_eye,
            scaleFactor= 1.5,
            minNeighbors=10,
            minSize=(5, 5),
        )
        
        if len(right_eye) > 0:
            biggest_right = right_eye[0]
            biggest_area = (biggest_right[2]-biggest_right[0])*(biggest_right[3]-biggest_right[1])
            if len(right_eye) > 1:
                for (ex, ey, ew, eh) in right_eye:
                    if (ew-ex)*(eh-ey) > biggest_area:
                        biggest_area = (ew-ex)*(eh-ey)
                        biggest_right = (ex, ey, ew, eh)
            (ex, ey, ew, eh) = biggest_right
            # cv2.rectangle(roi_color, (int(w/2) + ex, int(h*0.2) + ey), (int(w/2) + ex + ew, int(h*0.2) + ey + eh), (0, 255, 0), 2)
        
        # eyes = eyeCascade.detectMultiScale(
        #     roi_gray,
        #     scaleFactor= 1.5,
        #     minNeighbors=10,
        #     minSize=(5, 5),
        # )
        
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    
    img = cv2.flip(img, 1) # Flip camera horizontally
    cv2.imshow('video', img)
    # cv2.imshow('gray', gray)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
    elif count >= 100: # Take 100 face sample and stop video
        users.append(face_id)
        break
cap.release()
cv2.destroyAllWindows()