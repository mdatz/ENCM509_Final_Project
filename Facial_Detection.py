##ENCM 509 - Lab Project
##CNN Facial Recognizer

import cv2
import dlib
import scipy
import numpy
import os

print("Testing Image Read of .pgm file")
test = cv2.imread('./images/yaleB01/yaleB01_P00A+000E+00.pgm')

cv2.imshow('Test Image',test)
if cv2.waitKey():
    print("Success")
    exit()
