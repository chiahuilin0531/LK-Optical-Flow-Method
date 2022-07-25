"""
resource : https://github.com/opencv/opencv/blob/3.4/samples/python/tutorial_code/video/optical_flow/optical_flow.py
"""

from cProfile import label
from itertools import count
from turtle import st
from cv2 import waitKey
from matplotlib.colors import same_color
from matplotlib.pyplot import draw, gray, plot
import numpy as np
import cv2 as cv
import argparse
import imutils
import matplotlib.pyplot as plt
import pandas as pd
import pickle

NumOfDot = 30
Wid = 960

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

    def get_info(self):
        print(str(self.length) + '\t\t' + str(self.angle))
        # print(str(self.start) + '\t\t'  + str(self.stop) + '\t\t'  + str(self.vector) + '\t\t' + str(self.length) + '\t\t' + str(self.angle))

def checkInside(pt, mask = [], st = []):
    status = []
    for id in range(len(pt)):
        if st[id,0] == 0 or round(pt[id,0,1])>mask.shape[0] or round(pt[id,0,0])>mask.shape[1] :
            status.append([0])
        else:
            # print(st[id,0], round(pt[id,0,1]), round(pt[id,0,0]))
            status.append([mask[round(pt[id,0,1]), round(pt[id,0,0])] > 0])
    return np.array(status)

def modify_contrast_and_brightness2(img, brightness=0 , contrast=100):
    # 上面做法的問題：有做到對比增強，白的的確更白了。
    # 但沒有實現「黑的更黑」的效果
    import math

    # brightness = 0
    # contrast = 150 # - 減少對比度/+ 增加對比度

    B = brightness / 255.0
    c = contrast / 255.0 
    k = math.tan((45 + 44 * c) / 180 * math.pi)

    img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)

    # 所有值必須介於 0~255 之間，超過255 = 255，小於 0 = 0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img
    # print("減少對比度 (白黑都接近灰，分不清楚): ")
    # show_img(img)

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def process_img(frame, mask, border):
    test = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # print(test[0])
    # print(np.mean(test))
    # print(np.std(test))
    # test = np.float32(test)
    # test = (test - np.mean(test))/np.std(test)
    # test = test / 255
    # print(test[0])
    # test = test * (255/4) + (255/2)
    # print(test[0])
    # test = modify_contrast_and_brightness2(test)
    # test = np.clip(test, 0, 255)
    # test = np.uint8(test)
    # print(np.mean(test))
    # print(np.std(test))
    # print(test[30])
    # waitKey(0)
    # test = np.bitwise_and(test, mask)
    # avg_shi = np.round(np.mean(test[mask]), 2)
    # avg_shi = np.round(np.mean(test[border:]), 2)
    # print("\t\t" + str(avg_shi))
    # avg_shi = min(max(50, avg_shi), 100)

    # test = cv.GaussianBlur(test, (3, 3), 0)
    test = cv.threshold(test, np.mean(test) + np.std(test)*1.4, 255, cv.THRESH_BINARY)[1]
    # kernel = np.ones((3,3), np.uint8)
    test = cv.erode(test, None, iterations=1)
    # test = cv.dilate(test, None, iterations=1)
    return test
    
# Take first frame and find corners in it
def Run():
    ret, old_frame = cap.read()
    ratio = cap.get(cv.CAP_PROP_FRAME_HEIGHT)/cap.get(cv.CAP_PROP_FRAME_WIDTH)
    Hei = Wid*ratio
    old_frame = imutils.resize(old_frame, width=int(Wid))
    
    # Detection Bound
    bounds = dict(
        outerL= int(Wid*0.2),
        outerU= int(Hei*0.65),
        outerR= int(Wid*0.8),
        outerD= int(Hei*0.85),
        innerL= int(Wid*0.35),
        innerU= int(Hei*0.65),
        innerR= int(Wid*0.65),
        innerD= int(Hei*0.65),
    )

    mask_points = np.array([
                    # [bounds["outerL"], bounds["outerU"]], 
                    [bounds["outerL"], bounds["outerD"]], 
                    [bounds["outerR"], bounds["outerD"]], 
                    # [bounds["outerR"], bounds["outerU"]], 
                    [bounds["innerR"], bounds["innerU"]], 
                    # [bounds["innerR"], bounds["innerD"]], 
                    # [bounds["innerL"], bounds["innerD"]], 
                    [bounds["innerL"], bounds["innerU"]]])
    
    # cut a mask
    mask = np.zeros(old_frame.shape[:2], dtype=np.uint8)
    mask = cv.fillPoly(mask, [mask_points], 255)
                                        
    cv.imshow('frame', mask)
    waitKey(0)

    # old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    processed_old_frame = process_img(old_frame, mask, bounds["innerU"])
    
    # p0 = cv.goodFeaturesToTrack(old_gray, mask = mask, **feature_params)
    p0 = cv.goodFeaturesToTrack(processed_old_frame, mask = mask, **feature_params)

    # Create a mask image for drawing purposes
    draw_mask = np.zeros_like(old_frame)
    lines = []
    lengths = []
    count = 0

    while(1):
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            cv.destroyAllWindows()
            # exit()
            break

        frame = imutils.resize(frame, width=int(Wid))
        processed_frame = process_img(frame, mask, bounds["innerU"])
        # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # print(processed_old_frame.shape, processed_frame.shape)
        # calculate optical flow
        # p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        p1, st, err = cv.calcOpticalFlowPyrLK(processed_old_frame, processed_frame, p0, None, **lk_params)
        # print(processed_old_frame.shape, processed_frame.shape)
        # Select good points
        if p1 is not None:
            filter = checkInside(p1, mask, st)
            st[~filter] = 0
            good_new = p1[st==1]
            good_old = p0[st==1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            
            draw_mask = cv.line(draw_mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 8, color[i].tolist(), -1)
            # processed_frame = cv.circle(processed_frame, (int(a), int(b)), 8, color[i].tolist(), -1)
            
            new_line = Line([c, d], [a, b])
            lines.append(new_line)
            lengths.append(lines[-1].length)

        frame = cv.polylines(frame, [mask_points], True, (0, 0, 200), 3)
        # processed_frame = cv.polylines(processed_frame, [mask_points], True, (0, 0, 200), 3)
        

        img = cv.add(frame, draw_mask)
        cv.imshow('frame', img)

        processed_img = cv.add(cv.cvtColor(processed_frame, cv.COLOR_GRAY2BGR), draw_mask)
        cv.imshow('processed_frame', processed_img)
        

        # for ele in lines:
        #     ele.get_info()

        k = cv.waitKey(30) & 0xff
        if k == 27:
            cv.destroyAllWindows()
            break
        elif k == 32:
            while (1):
                k2 = cv.waitKey(0) & 0xff
                if k2 == 27:
                    cv.destroyAllWindows()
                    break
                elif k2 == 112:
                    print(processed_frame[0])
                elif k2 == 32:
                    break
        elif k == 8:
            draw_mask = np.zeros_like(old_frame)

        # Now update the previous frame and previous points
        processed_old_frame = processed_frame.copy()
        # old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        print(len(p0))

        if len(p0) < NumOfDot*0.2 or count == 50:
        # if count == 50:
            # print("\t", str(count))
            # if (count == 50): 
            count = 0
            new = cv.goodFeaturesToTrack(processed_old_frame, mask = mask, **feature_params)
            # new = cv.goodFeaturesToTrack(old_gray, mask = mask, **feature_params)
            if new is None:
                print("\t+ 0")
                continue
            
            print("\t+", str(len(new)))
            # p0 = new.reshape(-1, 1, 2)
            p0 = np.append(p0, new).reshape(-1, 1, 2)
            if (len(p0) > NumOfDot) :
                print("\t-", str(len(p0) - NumOfDot))
                p0 = p0[-NumOfDot:]

        count += 1
    
    save_object(lines, './line_segments.pkl')

    plt.title("Optical Flow Length Distribution")
    plt.xlabel("time")
    plt.ylabel("length of optical flow")
    plt.scatter(range(len(lengths)), lengths, 3)
    plt.show()

    plt.title("Optical Flow Length Frequency")
    plt.hist(lengths, label="frequency", bins=100)
    plt.xlabel("length of optical flow")
    plt.ylabel("frequency")
    plt.show()


Run()