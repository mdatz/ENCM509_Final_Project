##ENCM 509 - Lab Project
##CNN Facial Recognizer

import cv2
import dlib
import scipy
import numpy
import os

##Load Needed Face Detection Models
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

##Set Threshold Value for Matches
tolerance = 0.5

##Read Face Images
image = cv2.imread('./images/yaleB01/yaleB01_P00A+000E+00.pgm')

##Detect Face and Face Orientation/Shape
detected_faces = face_detector(image,1)
shapes_faces = [shape_predictor(image,face) for face in detected_faces]

##Determine Encoding For Given Face
encoding = [numpy.array(face_recognizer.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]
print(encoding)