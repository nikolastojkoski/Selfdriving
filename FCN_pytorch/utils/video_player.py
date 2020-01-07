import cv2
import numpy as np
import matplotlib.pyplot as plt

def play_video_file(video_file_path, output_folder_path='../output/',
                    out_filename='frame{}.png', out_size = (960, 640),
                    out_starting_idx=0, starting_frame_num = 1):
    """ Play a video file, and save screenshots to output folder"""

    cap = cv2.VideoCapture(video_file_path)
    curFrameNum = starting_frame_num
    playback_paused = False

    last_frame_num = 0
    while cap.isOpened():

        if curFrameNum != last_frame_num:
            ret, img = cap.read()
            if not ret:
                cap.set(1, curFrameNum+5)
                curFrameNum += 5
                print('.', end='')
                continue
        last_frame_num = curFrameNum

        img = cv2.resize(img, out_size)
        cv2.imshow('frame', img)

        keypress = cv2.waitKey(1)
        if keypress == ord('1'):
            cap.set(1, curFrameNum - 250)
            curFrameNum -= 500
        elif keypress == ord('3'):
            cap.set(1, curFrameNum + 250)
            curFrameNum += 500
        elif keypress == ord('2'):
            playback_paused = not playback_paused
        elif keypress == ord('s'):
            out_filepath = output_folder_path + out_filename.format(out_starting_idx)
            out_starting_idx += 1
            print('saved file:' + out_filepath)
            cv2.imwrite(out_filepath, img)
        elif keypress == ord('q'):
            break

        if not playback_paused:
            curFrameNum += 1

play_video_file('../sochi.mp4', '../output/frame_grab/', 'road_src_{}.png', out_starting_idx=170, starting_frame_num=2500)