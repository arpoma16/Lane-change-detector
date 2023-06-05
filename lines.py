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
        self.tracks = []
        self.x = l1
        self.y = l2
        self.tracks.append([self.x,self.y])
        self.done = False

        self.kfObj = KalmanFilter()
        self.predictedCoords = np.array([[l1],[l2]],dtype=np.float32)
        self.kfObj.first_detected(l1, l2)   

    def updateCoords(self, xn=None, yn=None):
        self.age = 0
        if xn is None or yn is None:
            return self.predict()
        self.tracks.append([xn,yn])
        self.x = xn
        self.y = yn
        self.predictedCoords = self.kfObj.Estimate(xn, yn)
        if len(self.tracks) >= 2:
            self.distl1 = self.tracks[-1][0] - self.tracks[-2][0] + self.distl1 
            self.distl2 = - self.tracks[-1][1] + self.tracks[-2][1] + self.distl2
            #print("id:" + str(self.i) + " disty : "+ str(self.disty))
        return self.predictedCoords
    
    def prediction(self):
        return self.predictedCoords

    def predict(self):
        self.predictedCoords = self.kfObj.predic()
        return self.predictedCoords
    def correct(self,x,y):
        return self.kfObj.correct(x,y)
    def setDone(self):
        self.done = True
    def timedOut(self):
        return self.done
    def age_one(self):
        self.age += 1
        if self.age > self.max_age:
            self.done = True
            #print('done')
        return True
    
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
        val = self.kf.predict()
        self.kf.correct(np.array([[coordX],[coordY]],dtype=np.float32))
        return val

    def correct(self, coordX, coordY):
        self.kf.correct(np.array([[coordX],[coordY]],dtype=np.float32))

    def predic(self):
        return self.kf.predict()
    
    def first_detected(self, coordX, coordY):
        self.kf.statePost=np.array([[coordX],[coordY],[0.],[0.]],dtype=np.float32)

            