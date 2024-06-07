#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:09:50 2024

@author: clement
"""

import numpy as np
import cv2
import glob
import time
import matplotlib.pyplot as plt
import random

#- Function to get only x% of calib pictures 
def sample_matching_elements(list1, list2, sample_size):
    indices = random.sample(range(len(list1)), sample_size)
    sampled_elements_list1 = [list1[i] for i in indices]
    sampled_elements_list2 = [list2[i] for i in indices]
    
    return sampled_elements_list1, sampled_elements_list2


# Critères de terminaison pour la précision des coins du damier
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Préparer les points d'objet (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
chessboardSize = (8,6)
frameSize = (800,600)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 30
objp = objp * size_of_chessboard_squares_mm


# Tableaux pour stocker les points d'objet et les points d'image des deux caméras
objpoints = [] # points 3D dans l'espace réel
imgpoints_left = [] # points 2D dans l'image de la caméra gauche
imgpoints_right = [] # points 2D dans l'image de la caméra droite

# Charger les images de calibration
images_left = sorted(glob.glob('left/*.jpg'))
images_right = sorted(glob.glob('right/*.jpg'))
import os
#images_left,images_right = sample_matching_elements(images_left, images_right, 30) 
for img_left, img_right in zip(images_left, images_right):
    imgL = cv2.imread(img_left)
    imgR = cv2.imread(img_right)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboardSize, None)
    #print(retL,retR)
    if retL and retR:
        objpoints.append(objp)

        corners2L = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpoints_left.append(corners2L)

        corners2R = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpoints_right.append(corners2R)

        cv2.drawChessboardCorners(imgL, (8,6), corners2L, retL)
        cv2.drawChessboardCorners(imgR, (8,6), corners2R, retR)
        cv2.imshow('imgL', imgL)
        cv2.imshow('imgR', imgR)
        res = cv2.waitKey(100000) & 0xFF
        if res == ord('s'):
            print('Save')
        elif res == ord('r'):
            print(img_left,img_right)   
            os.remove(img_left)
            os.remove(img_right)
    else:
            print(img_left,img_right)   
            os.remove(img_left)
            os.remove(img_right)        
#
cv2.destroyAllWindows()

# Calibration de chaque caméra
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
h,  w = imgL.shape[:2]
newCameraMatrixL, roi_L = cv2.getOptimalNewCameraMatrix(mtxL, distL, (w,h), 0, (w,h))


retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)
h,  w = imgR.shape[:2]
newCameraMatrixR, roi_R = cv2.getOptimalNewCameraMatrix(mtxR, distR, (w,h), 0, (w,h))


########## Stereo Vision Calibration #############################################

flags = 0
#flags |= cv2.CALIB_USE_INTRINSIC_GUESS 
#flags |= cv2.CALIB_USE_EXTRINSIC_GUESS 

criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retStereo, stereo_mtxL, stereo_distL, stereo_mtxR, stereo_distR, stereo_rot, stereo_trans, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)



########## Stereo Rectification #################################################
rectifyScale= 0
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv2.stereoRectify(stereo_mtxL, stereo_distL, stereo_mtxR, stereo_distR, grayL.shape[::-1], stereo_rot, stereo_trans)
stereoMapL = cv2.initUndistortRectifyMap(stereo_mtxL, stereo_distL, rectL, projMatrixL, grayL.shape[::-1], cv2.CV_16SC2)
stereoMapR = cv2.initUndistortRectifyMap(stereo_mtxR, stereo_distR, rectR, projMatrixR, grayL.shape[::-1], cv2.CV_16SC2)
stereoMapL_x = stereoMapL[0]
stereoMapL_y = stereoMapL[1]
stereoMapR_x = stereoMapR[0]
stereoMapR_y = stereoMapR[1]


np.savez('calibration/stereo_calib.npz', mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R1=projMatrixL, R2=projMatrixR, P1=projMatrixL, P2=projMatrixR, Q=Q,R=stereo_rot)

cv_file = cv2.FileStorage('calibration/stereoMap.xml', cv2.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x',stereoMapL[0])
cv_file.write('stereoMapL_y',stereoMapL[1])
cv_file.write('stereoMapR_x',stereoMapR[0])
cv_file.write('stereoMapR_y',stereoMapR[1])

cv_file.release()
#

