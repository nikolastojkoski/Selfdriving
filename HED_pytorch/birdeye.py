import cv2
import numpy as np
import math

def getBirdEyeView(image, trackbar_name):

    barsWindow = trackbar_name
    height = float(cv2.getTrackbarPos('height', barsWindow))
    theta01 = float(cv2.getTrackbarPos('theta.01', barsWindow))/100
    theta = (float(cv2.getTrackbarPos('theta', barsWindow)) + theta01 - 90) * math.pi/180
    focalLength = float(cv2.getTrackbarPos('f', barsWindow))
    pixelSkew = float(cv2.getTrackbarPos('pixel skew', barsWindow))
    h, w, ch = image.shape

    #projection matrix 2d->3d
    A1 = np.float32([[1, 0, -w/2],
                    [0, 1, -h/2],
                    [0, 0, 0],
                    [0, 0, 1]])

    R = np.float32([[1, 0, 0, 0],
                    [0, math.cos(theta), -math.sin(theta), 0],
                    [0, math.sin(theta), math.cos(theta), 0],
                    [0, 0, 0, 1]])
    #translation matrix
    T = np.float32([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, -height/(math.sin(theta) + 0.001)],
                    [0, 0, 0, 1]])
    #intristic matrix
    K = np.float32([[focalLength, pixelSkew, w/2, 0],
                    [0, focalLength, h/2, 0],
                    [0, 0, 1, 0]])

    transformationMat = np.dot(K, np.dot(T, np.dot(R, A1)))

    return cv2.warpPerspective(image, transformationMat, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC), transformationMat