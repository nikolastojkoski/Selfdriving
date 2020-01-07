import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from selfdriving import road_detection
from selfdriving import birdeye
from selfdriving import trajectory

def nothing(x):
    pass

#birdeyeview2 parameters
barsWindow2 = 'trackbar4'
cv2.namedWindow(barsWindow2, flags=cv2.WINDOW_FREERATIO)
cv2.namedWindow(barsWindow2, flags=cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('height', barsWindow2, 0, 2000, nothing)
cv2.createTrackbar('theta', barsWindow2, 0, 180, nothing)
cv2.createTrackbar('theta.01', barsWindow2, 0, 100, nothing)
cv2.createTrackbar('f', barsWindow2, 0, 2000, nothing)
cv2.createTrackbar('pixel skew', barsWindow2, 0, 100, nothing)
h,t,t01,f = 350, 7,0, 700
cv2.setTrackbarPos('height', barsWindow2, h)
cv2.setTrackbarPos('theta', barsWindow2, t)
cv2.setTrackbarPos('theta.01',barsWindow2,t01)
cv2.setTrackbarPos('f', barsWindow2, f)
cv2.setTrackbarPos('pixel skew', barsWindow2, 0)

#open video stream
curFrameNum = 500  #start moving
cap = cv2.VideoCapture('sochi2_3x.mp4')
cap.set(1, curFrameNum)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (960*2,640))

last_trajectory = None

while cap.isOpened():

    ret, image = cap.read()
    curFrameNum += 1

    # if frame is read correctly ret is True
    if not ret:
        print('.', end='')
        continue

    if curFrameNum > 6000:
        break

    print(curFrameNum)
    image = cv2.resize(image, (960, 640))
    cv2.imshow('frame', image)

    road_mask, combo_frame = road_detection.getRoadEdges(image)
    cv2.imshow('combo-frame',combo_frame)

    ipm_image, transformationMat = birdeye.getBirdEyeView(combo_frame, trackbar_name=barsWindow2)
    cv2.imshow('bird-eye-combo', ipm_image)
    
    road_mask_ipm = cv2.warpPerspective(road_mask, transformationMat, (960, 640), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
    cv2.imshow('road-mask-ipm', road_mask_ipm)
    
    trajectory_mask, combo_trajectory_image, last_trajectory = trajectory.findTrajectory(road_mask_ipm, ipm_image, last_trajectory)
    cv2.imshow('combo-ipm', combo_trajectory_image)
    
    unwarped_trajectory_mask = cv2.warpPerspective(trajectory_mask, transformationMat, (960, 640), flags=cv2.INTER_CUBIC)
    cv2.imshow('unwarped-trajectory', unwarped_trajectory_mask)
    
    unwarped_trajectory_mask_3ch = np.zeros_like(combo_frame)
    unwarped_trajectory_mask_3ch[:,:,2] = unwarped_trajectory_mask
    combo_frame = cv2.addWeighted(combo_frame, 1.0, unwarped_trajectory_mask_3ch, 1.0, 1.0)
    
    outframe = np.hstack((combo_frame, combo_trajectory_image))
    cv2.imshow('outframe', outframe)
    cv2.imshow('combo-trajectory-ipm', combo_trajectory_image)
    out.write(outframe)

    keypress = cv2.waitKey(1)
    if keypress == ord('q'):
       break
    elif keypress == ord('+'):
        curFrameNum += 250
        cap.set(1, curFrameNum)
    elif keypress == ord('1'):
        plt.figure()
        plt.imshow(combo_frame)
        plt.show()


out.release()
cap.release()
cv2.destroyAllWindows()