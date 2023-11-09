"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
import numpy as np
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

WIN = 'Example'

cv2.namedWindow(WIN)

while cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE):
    # We get a new frame from the webcam
    _, frame = webcam.read()
    og_frame = frame.copy()

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

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    #cv2.imshow("Demo", frame)


    height, width, num_channels = frame.shape
    image = np.empty((height, 2 * width, num_channels), frame.dtype)
    image[:, :width] = og_frame
    image[:, width:] = frame
    # if cv2.waitKey(1) == 32:
    #     image[:, width:] = og_frame
    cv2.imshow(WIN, image)

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()