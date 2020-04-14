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


##DataBase Values & Feature Vector List
db_count = 40
db_encodings = []
sample_start = 6                ##Fold I: 1  / Fold II: 6
num_samples = 5
query_sample_start = 1          ##Fold I: 6  / Fold II: 1
query_sample_end = 6            ##Fold I: 11 / Fold II: 6

##Read Images and Store Model Encodings into DataBase
for i in range(1, db_count):
    user_encodings = []
    
    ##Missing Training Sample :(
    if i == 14:
        db_encodings.append(user_encodings)
        continue
    
    j = sample_start
    while(len(user_encodings) != num_samples):
        ##Create File String & Read Image
        file_str = "./images/yaleB%02d/yaleB%02d (%d).pgm" % (i,i,j)
        print("Reading Image: " + file_str)
        image = cv2.imread(file_str)
        image = cv2.resize(image,(150,150))
        
        ##Detect Faces/Orientation and Determine Encoding
        encoding = [numpy.array(face_recognizer.compute_face_descriptor(image))]
		
        ##Output any failed db samples
        if encoding == []:
            print("Image %d Failed" % (i))
        else:  
            user_encodings.append(encoding)
        j += 1
    db_encodings.append(user_encodings)
print("DataBase Created!")

FA_count = 0
FR_count = 0
num_tests = 0
matched = 0
##how many ranks we will accept
accept_rank = 1
#for every user
for j in range(1, db_count):
    if j ==14:
        continue
    #for entries 6-10	
    for k in range(query_sample_start, query_sample_end):
        ##Test a Query Image Against DataBase Encodings
        query_file = "./images/yaleB%02d/yaleB%02d (%d).pgm" % (j,j,k)
        query_image = cv2.imread(query_file)
        query_image = cv2.resize(query_image,(150,150))
        print("Testing image %d from user %d" % (k,j))
        ##Get Query Image Encoding
        query_encoding = [numpy.array(face_recognizer.compute_face_descriptor(query_image))]
        if query_encoding == []:
            print("Query image failed to load")
            continue
        ##Check Difference from DB Images
		#for every user
        matching_scores = []
        for i in range(len(db_encodings)):
            if i == 14:
                continue
            matched = 0
		    #for every entry from that user
            for q in range(len(db_encodings[i])):
    
                diff = abs(numpy.subtract(query_encoding, db_encodings[i][q]))
                score = numpy.average(diff)
                user_score = (score, (i+1))
                matching_scores.append(user_score)
                
				
        num_tests+=1   
        dtype = [('score',float),('id',int)]
        allScores = numpy.array(matching_scores, dtype)
        ranked_scores = numpy.sort(allScores, order='score')
        correctMatch = 0
        falseMatch = 0		
        for z in range(0,accept_rank):
            print("Possible match with DB User #%d - Score:%.3f" % (ranked_scores[z][1],ranked_scores[z][0]))
            if ranked_scores[z][1] == j:
                correctMatch = 1
            else:
                falseMatch = 1
        if correctMatch != 1:
            FR_count+=1
        if falseMatch == 1:
            FA_count+=1		
        			
print("Total number of false rejections: %d" %(FR_count))
print("Total number of false acceptions: %d" %(FA_count))
print("FRR: %.4f" %(FR_count/num_tests))
print("FAR: %.4f" %(FA_count/num_tests))
