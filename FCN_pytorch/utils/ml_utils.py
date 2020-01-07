import cv2
import numpy as np
import math

cameraParameters = {'height':350.0,
                     'theta': 7.0,
                     'focalLength':700.0,
                     'pixelSkew': 1.0}

def getTransformationMatrix(camera_parameters, size=(960, 640)):

    height = camera_parameters['height']
    theta = (camera_parameters['theta'] - 90.0) * math.pi / 180
    focalLength = camera_parameters['focalLength']
    pixelSkew = camera_parameters['pixelSkew']
    w, h = size[0], size[1]

    A1 = np.float32([[1, 0, -w/2],
                    [0, 1, -h/2],
                    [0, 0, 0],
                    [0, 0, 1]])
    R = np.float32([[1, 0, 0, 0],
                    [0, math.cos(theta), -math.sin(theta), 0],
                    [0, math.sin(theta), math.cos(theta), 0],
                    [0, 0, 0, 1]])
    T = np.float32([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, -height/(math.sin(theta) + 0.001)],
                    [0, 0, 0, 1]])
    K = np.float32([[focalLength, pixelSkew, w/2, 0],
                    [0, focalLength, h/2, 0],
                    [0, 0, 1, 0]])

    transformationMat = np.dot(K, np.dot(T, np.dot(R, A1)))
    return transformationMat

def warpToBirdeye(image, transformationMat, dSize=(960, 640)):

    return cv2.warpPerspective(image, transformationMat, dSize,
                               flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)

def warpFromBirdeye(image, transformationMat, dSize=(960, 640)):

    return cv2.warpPerspective(image, transformationMat, dSize, flags=cv2.INTER_CUBIC)

def findTrajectory(road_mask, previous_trajectory=None):
    """ Calculates a trajectory following the middle of the road mask.
        Approximation is done with 3rd degree polynomial weighted least squares method.
        Weights are calculated based on previous_trajectory for time stability."""

    height, width = road_mask.shape
    trajectory_image = np.zeros((height, width, 1), dtype=np.uint8)
    rowcol = np.argwhere(road_mask)
    f = None
    if len(rowcol) != 0:

        if previous_trajectory != None:
            #weights for distance to previous_trajectory
            column_values = rowcol[:, 1]
            row_values = rowcol[:, 0]
            prev_column_values = previous_trajectory(row_values)
            distances = np.abs(column_values - prev_column_values)
            distances = distances / max(distances)
            weights = 1 - distances
            coefs = np.polyfit(rowcol[:, 0], rowcol[:, 1], 3, w=weights)
        else:
            coefs = np.polyfit(rowcol[:, 0], rowcol[:, 1], 3)

        f = np.poly1d(coefs)
        for row in range(min(rowcol[:, 0]), max(rowcol[:, 0])):
            col = int(f(row))
            if col > 0 and col < width:
                trajectory_image[row, int(f(row))] = 255

    return trajectory_image, f


def createTrajectoryVideo(inp_video_filepath, inp_road_mask_filepath, out_video_filepath,
                     size=(960,640), camera_parameters=cameraParameters):

    cap_video = cv2.VideoCapture(inp_video_filepath)
    cap_mask = cv2.VideoCapture(inp_road_mask_filepath)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_video_filepath, fourcc, 20.0, size)
    if not out.isOpened():
        print('cannot open video file for writing')
        return

    curFrameNum = 1
    last_trajectory = None
    while cap_video.isOpened():

        ret, image = cap_video.read()
        if not ret:
            break

        ret, mask = cap_mask.read()
        if not ret:
            break

        if curFrameNum % 100 == 0:
            print('cur_frame_num:{}'.format(curFrameNum))

        image = cv2.resize(image, size)
        transformationMat = getTransformationMatrix(camera_parameters)
        ipm_mask = warpToBirdeye(mask, transformationMat)
        ipm_trajectory_mask, last_trajectory = findTrajectory(ipm_mask, last_trajectory)
        trajectory_mask = warpFromBirdeye(ipm_trajectory_mask, transformationMat)

        trajectory_mask_3ch = np.zeros_like(image)
        trajectory_mask_3ch[:,:,2] = trajectory_mask
        combo_frame = cv2.addWeighted(image, 1.0, trajectory_mask_3ch, 1.0, 1.0)
        out.write(combo_frame)

    out.release()
    cap_video.release()
    cap_mask.release()


