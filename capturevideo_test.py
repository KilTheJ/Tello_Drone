# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 14:57:44 2025

@author: libra
"""

#import module for tello:
from djitellopy import tello

#Import Threading module:
import threading

#import opencv python module:
import cv2
import time 


#Global Variable
global img

#Start Connection With Drone
Drone = tello.Tello()
Drone.connect()

#Get Battery Info
def getTrame() : 
    while True : 
        print("Batterie : "  + str(Drone.query_battery()))
        print("Altitude : "  + str(Drone.query_attitude()))
    # print("Barométrie : " + str (Drone.query_barometer()))
    # # print("Température : " + str(Drone.query_temperature()))
    # print("Accel x : " + str(Drone.get_acceleration_x()))
    # print("Accel y : " + str(Drone.get_acceleration_y()))
    # print("Accel z : " + str(Drone.get_acceleration_z()))
    # # print("Roulis : " + str(Drone.get_roll()))
    # # print("Tangage : " + str(Drone.get_pitch()))
    # # print("Lacet : " + str(Drone.get_yaw()))
    time.sleep(0.033)


#Start Camera Display Stream
Drone.streamon()

t = threading.Thread(target=getTrame, daemon=True)
t.start()

while True:

# #Get Frame From Drone Camera Camera 
    img = Drone.get_frame_read().frame
    img = cv2.resize(img, (1080,720))
# #Show The Frame
    cv2.imshow("DroneCapture", img)
    cv2.waitKey(1)
    
