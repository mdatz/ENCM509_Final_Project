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

##DataBase Values
db_count = 20
db_encodings = []

##Read Images and Store Model Encodings into DataBase
for i in range(1, db_count):
    
    ##Missing Training Sample :(
    if i == 14:
        continue
        
    file_str = "./images/yaleB%02d/yaleB%02d_P00A+000E+00.pgm" % (i,i)
    print("Reading Image: " + file_str)
    image = cv2.imread(file_str)
    detected_faces = face_detector(image,1)
    shapes_faces = [shape_predictor(image,face) for face in detected_faces]
    
    encoding = [numpy.array(face_recognizer.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]
    db_encodings.append(encoding)

print("DataBase Created!")

##Test a Query Image Against DataBase Encodings
query_image = cv2.imread("./images/yaleB20/yaleB20_P00A+000E+00.pgm")