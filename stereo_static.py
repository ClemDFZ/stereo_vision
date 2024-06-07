#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:09:50 2024

@author: clement
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def overlay_images(image1, image2, alpha=0.5):
    # Vérification des dimensions des images
    if image1.shape != image2.shape:
        raise ValueError("Les dimensions des images ne correspondent pas.")

    # Superposition des images avec un facteur d'opacité alpha
    overlaid_image = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)

    return overlaid_image

def combine_images(image1, image2):
    # Vérifiez si les images ont la même hauteur
    if image1.shape[0] != image2.shape[0]:
        # Redimensionnez les images pour avoir la même hauteur
        height = min(image1.shape[0], image2.shape[0])
        image1 = cv2.resize(image1, (int(image1.shape[1] * height / image1.shape[0]), height))
        image2 = cv2.resize(image2, (int(image2.shape[1] * height / image2.shape[0]), height))
    
    # Concaténer les images horizontalement
    combined_image = np.hstack((image1, image2))
    return combined_image

with np.load('calibration/stereo_calib.npz') as data:
    mtxL = data['mtxL']
    distL = data['distL']
    mtxR = data['mtxR']
    distR = data['distR']
#    R = data['R']
#    T = data['T']
    R1 = data['R1']
    R2 = data['R2']
    projMatrixL = data['P1']
    projMatrixR = data['P2']
    Q = data['Q']

# Chargement des cartes de rectification
cv_file = cv2.FileStorage('calibration/stereoMap.xml', cv2.FILE_STORAGE_READ)
stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
cv_file.release()

##### WHEN USING THE CAMERA #####
#image_size = (800, 600)  # Taille de l'image
#cap_left = cv2.VideoCapture(0)
#cap_right = cv2.VideoCapture(4)
#for cap in [cap_left,cap_right]:
#    cap.set(3,800)
#    cap.set(4,600)
#
#ret_left, frame_left = cap_left.read()
#ret_right, frame_right = cap_right.read()
#base_image = combine_images(frame_left,frame_right)
#
#if not ret_left or not ret_right:
#    print("Erreur: Impossible de lire les images des caméras.")
##    break
##    else:   
#cap_left.release()
#cap_right.release() 
#####################################




frame_left = cv2.imread('result/left_image.jpg')

frame_right= cv2.imread('result/right_image.jpg')



img_undistorted_left = cv2.undistort(frame_left, mtxL, distL, None, newCameraMatrix=mtxL)
img_undistorted_right = cv2.undistort(frame_right, mtxR, distR, None, newCameraMatrix=mtxR)        
undistorted = combine_images(img_undistorted_left,img_undistorted_right)        

rectified_left = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,0)
rectified_right = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,0)   
rectified_pic = combine_images(rectified_left,rectified_right)

gray_left = cv2.cvtColor(rectified_left,cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(rectified_right,cv2.COLOR_BGR2GRAY)

overlayed_base = overlay_images(frame_left,frame_right)
overlayed_rectified = overlay_images(rectified_left,rectified_right)

overlayed_combined = combine_images(overlayed_base,overlayed_rectified)


stereo = cv2.StereoSGBM_create(numDisparities=272, blockSize=5)
disparity = stereo.compute(gray_left,gray_right)
depth = cv2.reprojectImageTo3D(disparity,Q)



while True:
    cv2.imshow("Only cam calib", undistorted)
    cv2.imshow("Rectified", rectified_pic) 
    cv2.imshow("Disparity Map", disparity)    
    cv2.imshow("Depth Map", depth)    
    cv2.imshow('Overlayed',overlayed_combined)
    # Attendre l'appui de la touche 'q' pour quitter la boucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#for [image,savename] in [[frame_left,"left_image"],
#                         [frame_right,"right_image"],
#                         [rectified_left,"rectified_left"],
#                         [rectified_right,"rectified_right"],
#                         [overlayed_combined,"overlayed_combined"],
#                         [disparity,"disparity"],
#                         [depth,"depth_map"]]:
#    cv2.imwrite("result/"+savename+".jpg",image)    
    
    
cv2.destroyAllWindows()