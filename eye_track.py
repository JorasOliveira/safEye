import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect_faces(img):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
    return frame

def detect_eyes(img):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_frame, 1.3, 5) # detect eyes
    width = np.size(img, 1) # get face frame width
    height = np.size(img, 0) # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
        else:
            right_eye = img[y:y + h, x:x + w]
    return left_eye, right_eye

img = cv2.imread("img/joras1.jpeg")
face = detect_faces(img)

left_eye, right_eye = detect_eyes(face)

threshold = 60
_, left_eye = cv2.threshold(cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY), threshold, 255, cv2.THRESH_BINARY)
_, right_eye = cv2.threshold(cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY), threshold, 255, cv2.THRESH_BINARY)

def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)
    return img

left_eye = cut_eyebrows(left_eye)
right_eye = cut_eyebrows(right_eye)

# def blob_process(img, detector):
#     gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, img = cv2.threshold(gray_frame, 42, 255, cv2.THRESH_BINARY)
#     keypoints = detector.detect(img)
#     return keypoints
# keypoints = blob_process(face, detect_eyes)
# cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('face',face)
cv2.imshow('left eye',left_eye)
cv2.imshow('right eye',right_eye)
cv2.waitKey(0)
cv2.destroyAllWindows()