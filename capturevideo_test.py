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

def getVideo():

    #Configuration de la video
    frame_width = 1080
    frame_height = 720
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter("output.mp4", fourcc, 60.0, (frame_width, frame_height))

    while True:
    # #Get Frame From Drone Camera Camera 
        img = Drone.get_frame_read().frame
        img = cv2.resize(img, (frame_width,frame_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #Sauvegarde de la frame 
        out.write(img)
    # #Show The Frame
        cv2.imshow('RGB Image',img )
        cv2.waitKey(1)
        
        if cv2.waitKey(1) == ord('q'):
            img.release()
            cv2.destroyAllWindows()
            break
        
#Start Camera Display Stream
Drone.streamon()

#Gestion du thread pour la vidéo
t = threading.Thread(target=getVideo, daemon=True)
t.start()
stop_thread=False


