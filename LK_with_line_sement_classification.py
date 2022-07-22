"""
resource : https://github.com/opencv/opencv/blob/3.4/samples/python/tutorial_code/video/optical_flow/optical_flow.py
"""

from cProfile import label
from itertools import count
from turtle import st
from cv2 import waitKey
from matplotlib.pyplot import draw, plot
import numpy as np
import cv2 as cv
import argparse
import imutils
import matplotlib.pyplot as plt
import pandas as pd

NumOfDot = 30
Wid = 960
# ratio = 9/16
# count = 0

# Hei = Wid*ratio

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
                    #    minDistance = 30,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


# Create some random colors
color = np.random.randint(0, 255, (NumOfDot, 3))

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) / np.pi * 180
    if v1[1] < 0:
        angle = 360 - angle

    return angle

# Line Segment Class
class Line:
    def __init__(self, start=[0, 0], stop=[0, 0]):
        self.start = np.array(np.multiply(start, [1, -1])) 
        self.stop = np.array(np.multiply(stop, [1, -1]))
        self.vector = np.subtract(self.stop, self.start)
        self.length = np.round(np.linalg.norm(self.vector), 2)
        self.angle = angle_between(self.vector, [1, 0])
        # print(str(self.angle) + '°')

    def get_info(self):
        print(str(self.length) + '\t\t' + str(self.angle))
        # print(str(self.start) + '\t\t'  + str(self.stop) + '\t\t'  + str(self.vector) + '\t\t' + str(self.length) + '\t\t' + str(self.angle))

def checkInside(pt, mask = []):
    status = []
    for id in range(len(pt)):
        # print(pt[id,0,0], pt[id,0,1])
        status.append([mask[int(pt[id,0,1]), int(pt[id,0,0])] > 0])
    
    return np.array(status)
    
# Take first frame and find corners in it
def Run():
    ret, old_frame = cap.read()
    ratio = cap.get(cv.CAP_PROP_FRAME_HEIGHT)/cap.get(cv.CAP_PROP_FRAME_WIDTH)
    Hei = Wid*ratio

    old_frame = imutils.resize(old_frame, width=int(Wid))
    # print(cap.get(cv.CAP_PROP_FRAME_HEIGHT), cap.get(cv.CAP_PROP_FRAME_WIDTH))
    # print(Hei, Wid)
    
    # Detection Bound
    bounds = dict(
        outerL= int(Wid*0.1),
        outerU= int(Hei*0.6),
        outerR= int(Wid*0.9),
        outerD= int(Hei*0.75),
        innerL= int(Wid*0.45),
        innerU= int(Hei*0.55),
        innerR= int(Wid*0.55),
        innerD= int(Hei*0.55),
    )
    
    # cut a mask
    mask = np.zeros(old_frame.shape[:2], dtype=np.uint8)
    mask = cv.fillPoly(mask, [np.array([[bounds["outerL"], bounds["outerU"]], 
                                        [bounds["outerL"], bounds["outerD"]], 
                                        [bounds["outerR"], bounds["outerD"]], 
                                        [bounds["outerR"], bounds["outerU"]], 
                                        [bounds["innerR"], bounds["innerU"]], 
                                        [bounds["innerL"], bounds["innerU"]]])], 255)
    # mask = cv.rectangle(mask, (bounds["outerL"], bounds["outerU"]), (bounds["outerR"], bounds["outerD"]), 255, -1)
    cv.imshow('frame', mask)
    waitKey(0)
    # adj_old_frame = np.zeros(((bounds["outerD"]-bounds["outerU"]), (bounds["outerR"]-bounds["outerL"]), 3), dtype=np.uint8)

    # for i in range(adj_old_frame.shape[0]):
    #     if i > bounds["innerU"]-bounds["outerU"] and i < bounds["innerD"]-bounds["outerU"]:
    #         adj_old_frame[i][: bounds["innerL"]-bounds["outerL"]] = old_frame[i+bounds["outerU"]][bounds["outerL"] : bounds["innerL"]]
    #         adj_old_frame[i][bounds["innerR"]-bounds["outerL"] :] = old_frame[i+bounds["outerU"]][bounds["innerR"] : bounds["outerR"]]
    #     else:
    #         adj_old_frame[i][:] = old_frame[i+bounds["outerU"]][bounds["outerL"] : bounds["outerR"]]

    # old_gray = cv.cvtColor(adj_old_frame, cv.COLOR_BGR2GRAY)

    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    print(old_gray.shape, mask.shape)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = mask, **feature_params)

    # Create a mask image for drawing purposes
    draw_mask = np.zeros_like(old_frame)
    lines = []
    lengths = []
    # count = 0

    while(1):
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            cv.destroyAllWindows()
            # exit()
            break

        frame = imutils.resize(frame, width=int(Wid))

        # adj_frame = np.zeros(((bounds["outerD"]-bounds["outerU"]), (bounds["outerR"]-bounds["outerL"]), 3), dtype=np.uint8)
    
        # for i in range(adj_frame.shape[0]):
        #     if i > bounds["innerU"]-bounds["outerU"] and i < bounds["innerD"]-bounds["outerU"]:
        #         adj_frame[i][: bounds["innerL"]-bounds["outerL"]] = frame[i+bounds["outerU"]][bounds["outerL"] : bounds["innerL"]]
        #         adj_frame[i][bounds["innerR"]-bounds["outerL"] :] = frame[i+bounds["outerU"]][bounds["innerR"] : bounds["outerR"]]
        #     else:
        #         adj_frame[i][:] = frame[i+bounds["outerU"]][bounds["outerL"] : bounds["outerR"]]

        # frame_gray = cv.cvtColor(adj_frame, cv.COLOR_BGR2GRAY)
        
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # print(p0)
        # print(p1)
        # print(st)
        # print("\n\n")

        # Select good points
        if p1 is not None:
            # st[~isInside(p1, bounds)] = 0
            # print(p0)
            # print(p1)
            # print(st)
            # print(st.shape)
            filter = checkInside(p1, mask)
            # print(filter)
            # print(filter.shape)
            st[~filter] = 0

            # st[p1[:,:,0] < bounds["outerL"]] = 0
            # st[p1[:,:,0] > bounds["outerR"]] = 0
            # st[p1[:,:,1] < bounds["outerU"]] = 0
            # st[p1[:,:,1] > bounds["outerD"]] = 0
            # print(st)
            good_new = p1[st==1]
            good_old = p0[st==1]

        # print(good_new, good_old)
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            # print((a, b), (c, d))
            draw_mask = cv.line(draw_mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            # draw_mask = cv.line(draw_mask, (int(a+bounds["outerL"]), int(b+bounds["outerU"])), (int(c+bounds["outerL"]), int(d+bounds["outerU"])), color[i].tolist(), 2)
            # frame = cv.circle(frame, (int(a+bounds["outerL"]), int(b+bounds["outerU"])), 5, color[i].tolist(), -1)

            new_line = Line([c, d], [a, b])
            lines.append(new_line)
            lengths.append(lines[-1].length)

        # frame = cv.rectangle(frame, (bounds["outerL"], bounds["outerU"]), (bounds["outerR"], bounds["outerD"]), (0, 200, 0), 3)
        frame = cv.polylines(frame,  [np.array([[bounds["outerL"], bounds["outerU"]], 
                                                [bounds["outerL"], bounds["outerD"]], 
                                                [bounds["outerR"], bounds["outerD"]], 
                                                [bounds["outerR"], bounds["outerU"]], 
                                                [bounds["innerR"], bounds["innerU"]], 
                                                [bounds["innerL"], bounds["innerU"]], 
                                                [bounds["outerL"], bounds["outerU"]]])], False, (0, 0, 200), 3)
        # frame = cv.rectangle(frame, (bounds["innerL"], bounds["innerU"]), (bounds["innerR"], bounds["innerD"]), (0, 0, 200), 3)
        img = cv.add(frame, draw_mask)
        
        cv.imshow('frame', img)
        # for ele in lines:
        #     ele.get_info()

        # k = cv.waitKey(0) & 0xff
        k = cv.waitKey(30) & 0xff
        if k == 27:
            cv.destroyAllWindows()
            # exit()
            break

        # Now update the previous frame and previous points
        
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        # count += 1
        print(len(p0))
        if len(p0) < NumOfDot/4 :
        # if count >= 30:
            # print(count)
            # count = 0
            new = cv.goodFeaturesToTrack(old_gray, mask = mask, **feature_params)
            if new is None:
                print("\t+ 0")
                continue
            
            print("\t+", str(len(new)))
            # print(new.shape, new)
            # print(p0.shape, p0)
            p0 = np.append(p0, new).reshape(-1, 1, 2)
            # print(p0.shape, p0)
            if (len(p0) > NumOfDot) :
                p0 = p0[-NumOfDot:]
    
    # d = []
    # for id, ele in enumerate(lines):
    #     d.append(
    #         {
    #             "time" : id,
    #             "length" : ele.length,
    #             "angle" : ele.angle
    #         }

    #     )
    # d = {}

    plt.title("Optical Flow Length Distribution")
    plt.xlabel("time")
    plt.ylabel("length of optical flow")
    plt.scatter(range(len(lengths)), lengths, 3)
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # for id, ele in enumerate(lines):
    #     if (ele.length < 20):
    #         plt.scatter(id, ele.length, s=1)
        # if ele.length in d:
        #     d[ele.length] += 1
        # else:
        #     d[ele.length] = 1
    plt.show()

    # print(lengths)

    plt.title("Optical Flow Length Frequency")
    plt.hist(lengths, label="frequency", bins=100)
    plt.xlabel("length of optical flow")
    plt.ylabel("frequency")
    plt.show()

    # for k in sorted(d):
    #     ax2.scatter(k, d[k])

    #     if (ele.angle < 90):
    #         ax1.scatter(id, ele.length, s=1, color='r')
    #     elif (ele.angle < 180):
    #         ax1.scatter(id, ele.length, s=1, color='y')
    #     elif (ele.angle < 270):
    #         ax1.scatter(id, ele.length, s=1, color='g')
    #     else:
    #         ax1.scatter(id, ele.length, s=1, color='b')


Run()