import cv2
import numpy as np
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import pyvirtualcam
import matplotlib.pyplot as plt

from pychubby.actions import Chubbify, Multiple, Pipeline, Smile
from pychubby.detect import LandmarkFace


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
with pyvirtualcam.Camera(width=500, height=500, fps=10, print_fps=True) as cam:
    while(True):
        blank_image = np.zeros((500, 500, 4), np.uint8)
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=500, height=500)
        img = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        image = frame

        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # show the face number
            cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 0, 255, 255), -1)
                cv2.circle(blank_image, (x, y), 2, (0, 0, 255, 0), -1)
        cv2.imshow('Video', blank_image)
        cv2.imshow('video1', image)

        # img = frame
        print(frame.shape)
        try:
            lf = LandmarkFace.estimate(img)

            a_per_face = Pipeline([Smile()])
            a_all = Multiple(a_per_face)

            new_lf, _ = a_all.perform(lf)
            new_img = new_lf
            #new_lf.plot(figsize=(5, 5), show_numbers=False)
        except:
            pass
        # data = np.fromstring(new_lf, dtype=np.uint8, sep='')
        # cv2.imshow('video', data)
        cam.send(blank_image)
        cam.sleep_until_next_frame()


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()