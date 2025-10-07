# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 08:40:41 2025

@author: libra
"""

import cv2
import numpy 

path = "output/"
f = open(path+"matrices.txt", "a")

vidcap = cv2.VideoCapture(path + 'output.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite(path + "frame%d.png" % count, image)
  success,image = vidcap.read()
  count += 1
  print('Frame :  ' ,count , success)
  #Conversion en matrice
for i in range(15) :
    # print("frame" + str(i) + ".png")
    img = cv2.imread(path+ "frame" + str(i) + ".png", 0)
    numpy.savetxt(f, img)
    
  
f.close()
print("Terminé")
