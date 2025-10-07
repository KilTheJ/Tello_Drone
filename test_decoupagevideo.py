# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 08:40:41 2025

@author: libra
"""

import cv2

vidcap = cv2.VideoCapture('output\output.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frame%d.png" % count, image)     # save frame as JPEG file      
  #Conversion en matrice
  img = cv2.imread("frame%d.png", 0)
  #TODO ENREGISTRER LES FICHIERS 
  success,image = vidcap.read()
  count += 1
  print('Frame :  ' ,count , success)
  
print("Terminé")