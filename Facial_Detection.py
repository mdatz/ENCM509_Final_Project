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
tolerance = 0.03

##DataBase Values & Feature Vector List
db_count = 39
db_encodings = []
num_samples = 10

##Read Images and Store Model Encodings into DataBase
for i in range(1, db_count):
    
    ##Missing Training Sample :(
    if i == 14:
        continue
        
    for j in range(1,num_samples):
        ##Create File String & Read Image
        file_str = "./images/yaleB%02d/yaleB%02d (%1d).pgm" % (i,i,j)
        print("Reading Image: " + file_str)
        image = cv2.imread(file_str)
		
        ##Detect Faces/Orientation and Determine Encoding
        detected_faces = face_detector(image,1)
        shapes_faces = [shape_predictor(image,face) for face in detected_faces]
        encoding = [numpy.array(face_recognizer.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]
		
        ##Output any failed db samples
        if encoding == []:
            print("Image %d Failed" % (i))
        else:  
            db_encodings.append(encoding)
print("DataBase Created!")

##Test a Query Image Against DataBase Encodings
query_image = cv2.imread("./images/yaleB03/yaleB03 (2).pgm")

##Get Query Image Encoding
detected_faces = face_detector(query_image,1)
shapes_faces = [shape_predictor(query_image,face) for face in detected_faces]
query_encoding = [numpy.array(face_recognizer.compute_face_descriptor(query_image, face_pose, 1)) for face_pose in shapes_faces]

##Check Difference from DB Images
for i in range(len(db_encodings)):
    
    diff = abs(numpy.subtract(query_encoding, db_encodings[i]))
    score = numpy.average(diff)
        
    if score <= tolerance:
        print("Possible match with DB Entry #%d - Score:%.3f" % ((i+1),score))