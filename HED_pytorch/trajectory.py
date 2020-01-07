import cv2
import numpy as np
import matplotlib.pyplot as plt

def findTrajectory(road_mask, frame, previous_trajectory=None):
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

        #for row in range(0, height):
        for row in range(min(rowcol[:, 0]), max(rowcol[:, 0])):
            col = int(f(row))
            if col > 0 and col < width:
                #trajectory_image[row, int(f(row))] = (255, 255, 255)
                trajectory_image[row, int(f(row))] = 255
                frame[row, int(f(row))] = (0, 0, 255)

        cv2.imshow('trajectory', trajectory_image)

    return trajectory_image, frame, f