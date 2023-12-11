import cv2
import numpy as np
from gaze_tracking import GazeTracking
from flask import Flask, render_template
from flask_sock import Sock

app = Flask(__name__)
sock = Sock(app)

gaze = GazeTracking()
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

@app.route("/")
def index():
    return render_template('example.html')

@sock.route('/socket')
def echo(socket):
    while True:
        input_data = socket.receive()
        input_array = np.frombuffer(input_data, np.uint8)
        input_image = cv2.imdecode(input_array, cv2.IMREAD_COLOR)
        output, text = process(input_image)
        socket.send(str.encode("0"+output))
        socket.send(str.encode("1"+text))

def process(image):
    names = ['None', 'Matheus']
    # Define min window size to be recognized as a face
    minW = 0.1*image.shape[1]
    minH = 0.1*image.shape[0]

    output = ""

    ## Facial Recognition

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
    )
    
    for(x,y,w,h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # If confidence is less then 100 ==> "0" : perfect match
        id = names[id] if confidence < 100 else "unknown"
        confidence = "  {0}%".format(round(100 - confidence))
        output += id + " confidence: " + confidence + "\n"
    
    ## Gaze Tracking

    gaze.refresh(image)

    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_up():
        text = "Looking up"
    elif gaze.is_down():
        text = "Looking down"
    elif gaze.is_center():
        text = "Looking center"
    else:
        text = "Undetected"

    # output += text + "\n"
    return output, text

if __name__ == "__main__":
    app.run(debug=True)