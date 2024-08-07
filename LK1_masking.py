"""
resource : https://github.com/opencv/opencv/blob/3.4/samples/python/tutorial_code/video/optical_flow/optical_flow.py
"""

from cv2 import waitKey
import numpy as np
import cv2 as cv
import argparse
import keyboard
import imutils

NumOfDot = 100
Wid = 960
ratio = 9/16
# count = 0

Hei = Wid*ratio

np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                              The example file can be downloaded from: \
                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()

cap = cv.VideoCapture(args.image)

# Check success
if not cap.isOpened():
    raise Exception("Could not open video device")

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = NumOfDot,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Detection Bound
bounds = dict(
    outerL= int(Wid*0.1),
    outerU= int(Hei*0.1),
    outerR= int(Wid*0.9),
    outerD= int(Hei*0.9),
    innerL= int(Wid*0.4),
    innerU= int(Hei*0.4),
    innerR= int(Wid*0.6),
    innerD= int(Hei*0.6),
)

# Create some random colors
color = np.random.randint(0, 255, (NumOfDot, 3))

# Take first frame and find corners in it
def Run():
    ret, old_frame = cap.read()

    old_frame = imutils.resize(old_frame, width=int(Wid))

    # mask = np.zeros_like(old_frame, dtype=np.uint8)
    # cv.rectangle(mask, (bounds["outerL"], bounds["outerU"]), (bounds["outerR"], bounds["outerD"]), (255, 255, 255), -1)
    # cv.rectangle(mask, (bounds["innerL"], bounds["innerU"]), (bounds["innerR"], bounds["innerD"]), (0, 0, 0), -1)
    # print(type(mask[100][200][1]))
    # cv.imshow("Rectangular Mask", mask)
    # waitKey(0)
    # cv.imshow('frame', masking)
    
    
    # cut a mask
    adj_old_frame = np.zeros(((bounds["outerD"]-bounds["outerU"]), (bounds["outerR"]-bounds["outerL"]), 3), dtype=np.uint8)

    for i in range(adj_old_frame.shape[0]):
        if i > bounds["innerU"]-bounds["outerU"] and i < bounds["innerD"]-bounds["outerU"]:
            adj_old_frame[i][: bounds["innerL"]-bounds["outerL"]] = old_frame[i+bounds["outerU"]][bounds["outerL"] : bounds["innerL"]]
            adj_old_frame[i][bounds["innerR"]-bounds["outerL"] :] = old_frame[i+bounds["outerU"]][bounds["innerR"] : bounds["outerR"]]
        else:
            adj_old_frame[i][:] = old_frame[i+bounds["outerU"]][bounds["outerL"] : bounds["outerR"]]
    # print(old_frame.shape, mask.shape)
    # adj_old_frame = cv.bitwise_and(old_frame, old_frame, mask=mask)
    # cv.imshow("Masked", adj_old_frame)
    # waitKey(0)

    old_gray = cv.cvtColor(adj_old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    draw_mask = np.zeros_like(old_frame)


    while(1):

        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            cv.destroyAllWindows()
            exit()
            break

        frame = imutils.resize(frame, width=int(Wid))
        # adj_frame = cv.bitwise_and(frame, frame, mask=mask)

        adj_frame = np.zeros(((bounds["outerD"]-bounds["outerU"]), (bounds["outerR"]-bounds["outerL"]), 3), dtype=np.uint8)
    
        for i in range(adj_frame.shape[0]):
            if i > bounds["innerU"]-bounds["outerU"] and i < bounds["innerD"]-bounds["outerU"]:
                adj_frame[i][: bounds["innerL"]-bounds["outerL"]] = frame[i+bounds["outerU"]][bounds["outerL"] : bounds["innerL"]]
                adj_frame[i][bounds["innerR"]-bounds["outerL"] :] = frame[i+bounds["outerU"]][bounds["innerR"] : bounds["outerR"]]
            else:
                adj_frame[i][:] = frame[i+bounds["outerU"]][bounds["outerL"] : bounds["outerR"]]

        frame_gray = cv.cvtColor(adj_frame, cv.COLOR_BGR2GRAY)
        # cv.imshow('frame', frame_gray)

        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            # draw_mask = cv.line(draw_mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            # frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            draw_mask = cv.line(draw_mask, (int(a+bounds["outerL"]), int(b+bounds["outerU"])), (int(c+bounds["outerL"]), int(d+bounds["outerU"])), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a+bounds["outerL"]), int(b+bounds["outerU"])), 5, color[i].tolist(), -1)

        frame = cv.rectangle(frame, (bounds["outerL"], bounds["outerU"]), (bounds["outerR"], bounds["outerD"]), (0, 200, 0), 3)
        frame = cv.rectangle(frame, (bounds["innerL"], bounds["innerU"]), (bounds["innerR"], bounds["innerD"]), (0, 0, 200), 3)
        img = cv.add(frame, draw_mask)
        
        cv.imshow('frame', img)

        k = cv.waitKey(30) & 0xff
        if k == 27:
            cv.destroyAllWindows()
            exit()
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        if (len(p1) < NumOfDot/4) :
            p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

Run()
