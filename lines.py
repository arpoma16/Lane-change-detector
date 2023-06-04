from random import randint
import time 
import cv2
import numpy as np 

# crea un objeto de tracking este objeto tiene una representacion  en x de cada linea.

class lanelines:
    def __init__(self,l1,l2,max_age):
        self.distl1=0
        self.distl2=0
        self.age = 0
        self.max_age = max_age
        self.state = '0'

        self.kfObj = KalmanFilter()
        self.predictedCoords = np.array([[l1],[l2]],dtype=np.float32)
        self.kfObj.first_detected(l1, l2)
        self.tracker =cv2.TrackerMIL_create()     

# Instantiate OCV kalman filter
class KalmanFilter:
    def __init__(self):
        q = 1e-5   #  process noise covariance
        r = 0.01 #  measurement noise covariance, r = 1
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov     = q* np.eye(4, dtype=np.float32)   # Q         
        self.kf.measurementNoiseCov = r* np.eye(2, dtype=np.float32)   # R
        self.kf.errorCovPost  = np.eye(4, dtype=np.float32)            # P0 = I
        #self.kf.errorCovPost = np.ones((1, 1))
        #self.kf.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        self.kf.correct(np.array([[coordX],[coordY]],dtype=np.float32))
        return self.kf.predict()

    def correct(self, coordX, coordY):
        self.kf.correct(np.array([[coordX],[coordY]],dtype=np.float32))

    def predic(self):
        return self.kf.predict()
    
    def first_detected(self, coordX, coordY):
        self.kf.statePost=np.array([[coordX],[coordY],[0.],[0.]],dtype=np.float32)

            