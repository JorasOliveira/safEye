import cv2
import numpy as np
import os
import json

# Path for face image database
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

users = []
with open("users.json") as json_file:
    users = json.load(json_file)
face_id = input("\nEnter user id and press <return> ==>  ")
if face_id in users["users"]:
    print(f"\nUser {face_id} already exists")
    exit()
print("\n[INFO] Initializing face capture. Look the camera and wait ...")
count = 0 # Initialize individual sampling face count

while(True):
    ret, img = cap.read()
    face_region = img[80:400, 160:480]
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
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
    
    img = cv2.flip(img, 1) # Flip camera horizontally
    cv2.imshow('video', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
    elif count >= 30: # Take 100 face sample and stop video
        users["users"].append(face_id)
        with open("users.json", "w") as json_file:
            json.dump(users, json_file)
        break
cap.release()
cv2.destroyAllWindows()

# function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img)
        for (x,y,w,h) in faces:
            faceSamples.append(img[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
# Save the model into trainer.yml
recognizer.write('trainer.yml') 
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))