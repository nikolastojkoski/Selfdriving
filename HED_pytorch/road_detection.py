import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import PIL
import time
from selfdriving import HED
import skimage.measure

def getEdgeMap(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = PIL.Image.fromarray(image)
    #image = image.resize((480, 320))

    tensorInput = torch.FloatTensor(np.array(image)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))
    tensorOutput = HED.estimate(tensorInput)
    edgeMap = np.array((tensorOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(np.uint8))

    return edgeMap

def getEdgeMap2x2(image):
    '''detect edges with HED, by resizing image to 960x640 and
    run hed on 4 seperate 480x320 windows and then stitch everything back to 960x640 '''

    w, h = 960, 640
    image_resized = cv2.resize(image, (960, 640))

    pad = 32
    img00 = image_resized[pad:320+pad, pad:480+pad]
    img01 = image_resized[pad:320+pad, 480-pad:960-pad]
    img10 = image_resized[320-pad:640-pad, pad:480+pad]
    img11 = image_resized[320-pad:640-pad, 480-pad:960-pad]

    edgemap00 = np.zeros((h-2*pad, w-2*pad), np.uint8)
    edgemap01 = np.zeros((h-2*pad, w-2*pad), np.uint8)
    edgemap10 = np.zeros((h-2*pad, w-2*pad), np.uint8)
    edgemap11 = np.zeros((h-2*pad, w-2*pad), np.uint8)

    ww, hh = 480, 320

    edgemap00[0:hh,0:ww] = cv2.copyTo(getEdgeMap(img00), mask=None)
    edgemap01[0:hh, ww-2*pad:] = cv2.copyTo(getEdgeMap(img01), mask=None)
    edgemap10[hh-2*pad:, 0:ww] = cv2.copyTo(getEdgeMap(img10), mask=None)
    edgemap11[hh-2*pad:, ww-2*pad:] = cv2.copyTo(getEdgeMap(img11), mask=None)

    e0 = cv2.max(edgemap00, edgemap01)
    e1 = cv2.max(edgemap10, edgemap11)
    edgemap = cv2.max(e0, e1)

    edgemap_resized = cv2.resize(edgemap, (960, 640))
    return edgemap_resized

def getEdgeMap_Sobel(image):

    h, w, _ = image.shape
    image = cv2.GaussianBlur(image, ksize=(3,3), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    scale, delta, ddepth = 1, 0, cv2.CV_16S
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta,borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta,borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

def getRoadEdges(image, fromBirdEye=False):

    #get frame edgemap
    edgeMapFrame = getEdgeMap2x2(image)
    #edgeMapFrame = getEdgeMap_Sobel(image)
    #cv2.imshow('HED-frame', edgeMapFrame)

    #apply thresh to get binary image
    ret, edges_thresh = cv2.threshold(edgeMapFrame, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)    #for normal view
    #ret, edges_thresh = cv2.threshold(edgeMapFrame, 20, 255, cv2.THRESH_BINARY)                       #for bird eye view

    #draw horizontal line at horizon (for correct floodfill)
    if fromBirdEye == True:
        edges_thresh[40, :] = 255 # for 960x640 bird-eye view
    else:
        cv2.rectangle(edges_thresh, (10, 275), (edges_thresh.shape[1]-10, edges_thresh.shape[0]-10), 255, 1) # for 960x640 normal view
    #cv2.imshow('HED-frame-thresh', edges_thresh)

    #get inverse binary edgemap
    edges_thresh_inv = cv2.bitwise_not(edges_thresh)
    #cv2.imshow('HED-frame-thresh-inv', edges_thresh_inv)

    #floodfill
    h_, w_ = edges_thresh.shape
    mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
    seed_point_x, seed_point_y = int(w_/2), 430    #for normal frame 960x640
    if fromBirdEye == True:
        seed_point_x, seed_point_y = 480, 610
    ret, road_filled, mask, rect = cv2.floodFill(edges_thresh, mask_, (seed_point_x ,seed_point_y), 255)
    #cv2.imshow('road_filled', road_filled)

    seed_point2 = (seed_point_x - 65, seed_point_y + 100)
    if road_filled[seed_point2[1], seed_point2[0]] != 255:
        ret, road_filled, mask, rect = cv2.floodFill(edges_thresh, mask_, seed_point2, 255)

    seed_point3 = (seed_point_x + 65, seed_point_y + 100)
    if road_filled[seed_point3[1], seed_point3[0]] != 255:
        ret, road_filled, mask, rect = cv2.floodFill(edges_thresh, mask_, seed_point3, 255)

    #bitwise and to extract road area
    road_mask = cv2.bitwise_and(road_filled, edges_thresh_inv)
    #cv2.imshow('road-mask', road_mask)

    #draw road area on frame
    image_resized = cv2.resize(image, (w_, h_))
    green_frame = np.zeros((h_, w_, 3), np.uint8)
    green_frame[:] = (0, 255, 0)
    road_area_green = cv2.copyTo(green_frame, road_mask)
    combo_frame = cv2.addWeighted(image_resized, 1, road_area_green, 0.35, 0.35)
    #cv2.imshow('combo-frame', combo_frame)

    cv2.drawMarker(combo_frame, (seed_point_x, seed_point_y), color=(0, 0, 255))
    cv2.drawMarker(combo_frame, seed_point2, color=(0, 0, 255))
    cv2.drawMarker(combo_frame, seed_point3, color=(0, 0, 255))

    #return road_mask, combo_frame
    return road_mask, edgeMapFrame, combo_frame

