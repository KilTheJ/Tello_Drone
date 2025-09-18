# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 14:57:44 2025

@author: libra
"""

#import module for tello:
from djitellopy import tello

#import opencv python module:
import cv2
#Global Variable
global img

#Start Connection With Drone
Drone = tello.Tello()
Drone.connect()

#Get Battery Info
print(Drone.get_battery())

#Start Camera Display Stream
Drone.streamon()
while True:

#Get Frame From Drone Camera Camera 
    img = Drone.get_frame_read().frame
    img = cv2.resize(img, (1080,720))
#Show The Frame
    cv2.imshow("DroneCapture", img)
    cv2.waitKey(1)