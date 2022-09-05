"""
resource : https://github.com/opencv/opencv/blob/3.4/samples/python/tutorial_code/video/optical_flow/optical_flow.py
"""

from cmath import nan
from copy import deepcopy
import csv
import itertools
from math import floor
import sys
import time
import numpy as np
import cv2 as cv
import argparse
import imutils
import matplotlib
import matplotlib.pyplot as plt
import pickle
from scipy import stats

# Hyper-parameters
WID = 860               
"window width"
Hei = 540               
"window height"
TP_NUM = 20             
"maximum tracking points at the same time"
VP_REF_NUM = 10
"number of recent cross points references to update the VP per round"
VP_UPDATE_RATE = 0.5
"update rate to the VP"
FL_UPDATE_RATE = 0.05
"update rate for calculating average length of flow lines"
TP_UPDATE_RATE = 0.3
"update rate tracking points when they are not enough"
TP_UPDATE_TIME = 10
"number of frames to update tracking points "
MIN_ANG_DIF = 25
"(degree) max acceptable angle difference of 2 lines to construct cross pt"
MAX_CP_STD = 1.0
"max acceptable standard deviation range of new cross points (distance between cross point and VP)"
MIN_FL_LEN = 1.0
"shortest acceptable length of flow lines"
CP_THOLD = 1/15
"max distance acceptable from new cross point to the VP (proportion of the window)"
HIDE_VP_THOLD = 50
"the number of frames that VP has not updated to reset VP"
FL_UPD_METH = "REP"
"EXTend flow points or REPlace them by new points"
SHOW_VL = 0
"0/no show, 1/show VL on frame, 2/show VL on both frame and plot"
VP_REF = 300
"Number of referenced VPs in VP history (0 for all vps)"
WRITE_VIDEO = True
"Output Video"
SHOW_DNMC_PLOT = False
"Show the dynamic plot of VP & cross points"


def setup():
    """Setup global variables"""
    global cap, feature_params, lk_params, color, video_name

    if SHOW_DNMC_PLOT:
        # matplotlib.use("Qt5agg") # or "Qt5agg" depending on you version of Qt
        fig = plt.figure(figsize=(12, 8), dpi=80)
        plt.title(f"Recent {VP_REF} Points")
        plt.gca().invert_yaxis()
        plt.axis('scaled')
        plt.xlim(WID//3*1, WID//3*2)
        plt.ylim(Hei//3, Hei//4*3)
        plt.ylabel("y axis")
        plt.xlabel("x axis")
        plt.show(block=False)

    np.set_printoptions(threshold=np.inf)
    parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                                The example file can be downloaded from: \
                                                https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
    parser.add_argument('image', type=str, help='path to image file')
    args = parser.parse_args()
    video_name = sys.argv[-1].split("\\")[-1].split(".")[0]
    cap = cv.VideoCapture(args.image)

    # Check success
    if not cap.isOpened():
        raise Exception("Could not open video device")

    # Step 0: Setup the Parameters

    # params for ShiTomasi corner detection
    feature_params = dict(  maxCorners = int(TP_NUM/4),
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (TP_NUM, 3))

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

class Point:
    """
    Vanishing Point/Cross Point Class
    =====
    A class to represent a vanishing point or cross point.

    Attributes
    ----------
    x : float
        x coordinate value
    y : float
        y coordinate value

    Methods
    -------
    is_init():
        Check whether the point is initialized. (only for the vanishing point)

    has_moved():
        Check whether the point has moved.

    set(x, y):
        Initialize the point with coordinate values.

    update(x, y):
        Update the point with movement coordinate values.

    check_valid(x, y):
        Check whether the new cross point is too far from the current vanishing point
    """
    def __init__(self, isVP: bool, x=None, y=None) -> None:
        self._isinit = not isVP
        self._moved = False
        self.x = x
        self.y = y
        pass

    def __sub__(self, other):
        return np.array([self.x-other.x, self.y-other.y])

    def __call__(self) -> tuple:
        return (self.x, self.y)

    def set(self, x, y):
        """Initialize the point with coordinate values."""
        self._isinit = True
        self.x = x
        self.y = y
        print("VP init")

    def update(self, x, y):
        """Update the point with movement coordinate values."""
        if not self._isinit:
            raise Exception("VP is not initialized")
        self._moved = True
        self.x = self.x + x * VP_UPDATE_RATE
        self.y = self.y + y * VP_UPDATE_RATE
        print("VP updated")

    def is_init(self) -> bool:
        return self._isinit

    def has_moved(self) -> bool:
        return self._moved

    def check_valid(self, x, y) -> bool:
        """Check whether the new cross point is too far from the current vanishing point"""
        return (np.abs(np.array([self.x-x, self.y-y])) < np.array([WID*CP_THOLD, Hei*CP_THOLD])).all()

class VL:
    """
    Vanishing Line Class
    =====
    A class to represent a pair of vanishing lines.

    Attributes
    ----------
    None

    Methods
    -------
    update(vps: list, best_vp: VP):
        Update the line pairs with a list of VP history and the current vanishing point.
    """
    def __init__(self):
        self._isinit = False

    def __call__(self, mode = 'best_point'):
        if not self._isinit:
            return None, None, None, None, None
        if mode == 'best_point':
            st = self._intercept is not nan and self._interceptv is not nan
            return self._lp, self._rp, self._up, self._dp, st
        else :
            return self._calculate_endpt()

    def update(self, vps: list, best_vp: Point):
        """Update the line pairs with a list of VP history and the current vanishing point."""
        if best_vp.has_moved():
            self._isinit = True
            self._bp = (best_vp.x, best_vp.y)
            x = [row.x for row in vps]
            y = [row.y for row in vps]

            slope, intercept, r, p, std_err = stats.linregress(x, y)
            self._m = slope
            self._intercept = intercept

            slope, intercept, r, p, std_err = stats.linregress(y, x)
            self._mv = slope
            self._interceptv = intercept

            self._lp = (0, self._bp[1] - self._bp[0]*self._m)
            self._rp = (WID-1, self._bp[1] + (WID - 1 - self._bp[0])*self._m)
            self._up = (self._bp[0] - self._bp[1]*self._mv, 0)
            self._dp = (self._bp[0] + (Hei-1 - self._bp[1])*self._mv, Hei-1)

    def _calculate_endpt(self):
        l = (0, self._intercept)
        r = (WID-1, self._intercept + (WID-1)*self._m)
        u = (self._interceptv, 0)
        d = (self._interceptv + (Hei-1)*self._mv, Hei-1)
        st = self._intercept is not nan and self._interceptv is not nan
        return l, r, u, d, st

class FlowLine:
    """
    Optical Flow Line Segment Class
    =====
    A class to represent a optical flow line.

    Attributes
    ----------
    start : array
        Start point.
    stop : array
        End point.
    angle : float
        Angle on the xy-coordinate system of the line.
    color : array
        BRG value of the line color.

    Methods
    -------
    get_info(vps: list, best_vp: VP):
        Print the information of the line.
    length():
        Return the length of the line.
    """
    def __init__(self, start=[0, 0], stop=[0, 0], color=[0,0,0]):
        self.start = np.array(start) 
        self.stop = np.array(stop)
        self._vector = np.subtract(np.multiply(self.stop, [1, -1]), 
                                np.multiply(self.start, [1, -1]))
        self._len = np.round(np.linalg.norm(self._vector), 2)
        self.angle = angle_between(self._vector, [1, 0])
        self.color = color

    def length(self):
        return self._len

    def get_info(self):
        """Print the information of the line."""
        # print(self.start)
        # print(str(self.length) + '\t\t' + str(self.angle))
        # print(str(self.start) + '\t\t'  + str(self.stop) + '\t\t'  + str(self.vector) + '\t\t' + str(self.length) + '\t\t' + str(self.angle))
        print(str(self.start) + '\t\t'  + str(self.stop) + '\t\t'  + str(np.concatenate([self.start, self.stop])))
        pass

def cross_point(line1, line2):
    """calculate cross point of 2 lines"""
    
    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    k1 = (y2-y1)*1.0/(x2-x1)
    b1 = y1*1.0-x1*k1*1.0
    if (x4-x3) == 0:
        k2 = None
        b2 = 0
    else:
        k2 = (y4-y3)*1.0/(x4-x3)
        b2 = y3*1.0-x3*k2*1.0
    if k2 == None:
        x = x3
    elif k1-k2 == 0:
        return[nan, nan]
    else:
        x = (b2-b1)*1.0/(k1-k2)
    y = k1*x*1.0+b1*1.0
    return [x, y]

def checkInside(pts, mask = [], st = []):
    """
        Brief
        ---
        Check whether the tracking point is still inside the ROI.

        Parameters
        ---
        mask : array
            A list of coordinates of the boundary points.
        st : array
            A list of current status.

        Returns
        ---
        isInside : boolean
    """
    status = []
    for id in range(len(pts)):
        if st[id,0] == 0 or floor(pts[id,0,1])>mask.shape[0] or floor(pts[id,0,0])>mask.shape[1] :
            status.append([0])
        else:
            status.append([mask[floor(pts[id,0,1]), floor(pts[id,0,0])] > 0])
    return np.array(status)

def modify_contrast_and_brightness(img, brightness=0 , contrast=100):
    """
        Brief
        ---
        Modify contrast and brightness of the image.

        Parameters
        ---
        brightness : int

        contrast : int
            Positive value for increase contrast, negative for decrease contrast.

        Returns
        ---
        img : array
            A modified 2D image.
    """
    import math

    B = brightness / 255.0
    c = contrast / 255.0 
    k = math.tan((45 + 44 * c) / 180 * math.pi)

    img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def read_object(filename):
    with open(filename, 'rb') as inp:  # Overwrites any existing file.
        lines = pickle.load(inp)
    return lines

def save_csv(list, filename):
    with open(f'./vps/vps_{filename}.csv', 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow(["x", "y"])
        write.writerows(list)

def read_csv(filename):
    x, y = [], []
    with open(f'./vps/vps_{filename}.csv') as csvfile:
        rows = csv.reader(csvfile)
        next(rows, None)
        for row in rows:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return x, y

def process_img(img):
    """Process the image so as to do the feature tracking."""
    processed_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # test = np.float32(test)
    # test = (test - np.mean(test))/np.std(test)
    # test = test / 255
    # test = test * (255/4) + (255/2)
    # processed_img = modify_contrast_and_brightness(processed_img)
    # test = np.clip(test, 0, 255)
    # test = np.uint8(test)
    # test[mask == 0] = 0
    # test = np.bitwise_and(test, mask)
    # avg_shi = np.round(np.mean(test[mask]), 2)
    # avg_shi = np.round(np.mean(test[border:]), 2)
    # avg_shi = min(max(50, avg_shi), 100)
    # processed_img = cv.GaussianBlur(processed_img, (5, 5), 0)
    processed_img = cv.GaussianBlur(processed_img, (3, 3), 0)
    # test = cv.threshold(test, np.mean(test[test!=0]) + np.std(test[test!=0])*1.5, 255, cv.THRESH_BINARY)[1]
    # kernel = np.ones((3,3), np.uint8)
    # test = cv.erode(test, None, iterations=1)
    # test = cv.dilate(test, None, iterations=1)
    return processed_img
    
def Run():
    # Step 1: Take first frame and find corners.
    ret, old_frame = cap.read()
    ratio = cap.get(cv.CAP_PROP_FRAME_HEIGHT)/cap.get(cv.CAP_PROP_FRAME_WIDTH)
    global Hei
    Hei = int(WID*ratio)
    old_frame = imutils.resize(old_frame, width=int(WID))     
    center = (int(WID/2), int(Hei/2))

    # if you want to save the video with VP & flow lines
    if WRITE_VIDEO:
        out = cv.VideoWriter(f'./saved_video/{video_name}.avi', 
            cv.VideoWriter_fourcc(*'MJPG'), 30, (WID,  Hei))
    
    # ROI Boundary
    bounds = dict(
        outerL= int(WID*0.2),
        outerU= int(Hei*0.65),
        outerR= int(WID*0.8),
        outerD= int(Hei*0.8),
        innerL= int(WID*0.47),
        innerU= int(Hei*0.65),
        innerR= int(WID*0.52),
        innerD= int(Hei*0.65),
    )


    # bounds = dict(
    #     outerL= int(WID*0.1),
    #     outerR= int(WID*0.9),
    #     outerD= int(Hei*0.9),
    #     innerL= int(WID*0.3),
    #     innerU= int(Hei*0.5),
    #     innerR= int(WID*0.7),
    # )

    mask_points = np.array([[ WID//2,                                   (bounds["outerD"]+bounds["innerU"])//2],    #中間 [0]
                            [ bounds["outerL"],                          bounds["outerD"]],                         #左下 [1]
                            [ WID//2,                                    bounds["outerD"]],                         #中下 [2]
                            [ bounds["outerR"],                          bounds["outerD"]],                         #右下 [3]
                            [(bounds["outerR"]+bounds["innerR"])//2,    (bounds["outerD"]+bounds["innerU"])//2],    #右中 [4]
                            [ bounds["innerR"],                          bounds["innerU"]],                         #右上 [5]
                            [ WID//2,                                    bounds["innerU"]],                         #中上 [6]
                            [ bounds["innerL"],                          bounds["innerU"]],                         #左上 [7]
                            [(bounds["outerL"]+bounds["innerL"])//2,    (bounds["outerD"]+bounds["innerU"])//2]])   #左中 [8]
    
    mask = np.zeros(old_frame.shape[:2], dtype=np.uint8)
    mask = cv.fillPoly(mask, [np.array([mask_points[1], mask_points[3], mask_points[5], mask_points[7]])], 255)
                        
    # Full ROI Schematic                
    cv.imshow('frame', mask)
    cv.waitKey(200)

    small_mask = []
    for i in range(4):
        small_mask.append(np.zeros_like(mask))

    small_mask[0] = cv.fillPoly(small_mask[0], [np.array([mask_points[0], mask_points[8], mask_points[1], mask_points[2]])], 255)
    small_mask[1] = cv.fillPoly(small_mask[1], [np.array([mask_points[0], mask_points[2], mask_points[3], mask_points[4]])], 255)
    small_mask[2] = cv.fillPoly(small_mask[2], [np.array([mask_points[0], mask_points[4], mask_points[5], mask_points[6]])], 255)
    small_mask[3] = cv.fillPoly(small_mask[3], [np.array([mask_points[0], mask_points[6], mask_points[7], mask_points[8]])], 255)

    # ROI Schematic Animation
    for i in range(4):
        cv.imshow('frame', small_mask[i])
        cv.waitKey(100)

    processed_old_frame = process_img(old_frame)

    # Step 2: Choose the points to track in the first round.
    p0s = []    
    # p0s is a list of 2 p0's, upper and lower part respectively.
    # p0 is a numpy array that can be sent into optical flow function.
    for j in range(2):
        p0 = []
        for i in range(2):
            new = cv.goodFeaturesToTrack(processed_old_frame, mask = small_mask[j*2+i], **feature_params)
            if new is not None:
                p0.extend(new)
        p0 = np.array(p0, dtype=np.float32)
        p0s.append(p0)
    
    vp_history_xy, all_vp = [], []
    recent_cps, all_cps = [], []
    flow_lines = []
    vl = VL()
    vp = Point(isVP=True)
    prev_time = time.time()
    avg_len = [MIN_FL_LEN, MIN_FL_LEN]

    vp_ult = 0  # number of frames passed since last time VP updated
    tp_ult = 0  # number of frames passed since last time tracking point updated
    draw_mask = np.zeros_like(old_frame) # draw the flow lines

    # Step 3: Start reading frames from videos
    while(1):
        ret, frame = cap.read()

        # Step 4: Check whether it is the end of the video
        if not ret:
            print('No frames grabbed!')
            cv.destroyAllWindows()
            break
        
        frame = imutils.resize(frame, width=int(WID))
        processed_frame = process_img(frame)

        total_lines = [] # in one frame
        
        # Step 5: Tracking Points Detected
        if len(p0s[0]) != 0 or len(p0s[1]) != 0:
            for n, p0 in enumerate(p0s):
                if len(p0) == 0:
                    continue
                good_new = []
                good_old = []
                cur_lines = []
                # calculate optical flow
                p1, st, err = cv.calcOpticalFlowPyrLK(processed_old_frame, 
                                processed_frame, p0, None, **lk_params)
            
                # Step 6: Check whether the flow lines are valid (to generate cross points)

                # Step 6-1: Check if the tracked points are still inside the mask
                if p1 is not None:
                    filter = checkInside(p1, mask, st)
                    st[~filter] = 0
                    good_new.extend(p1[st==1])
                    good_old.extend(p0[st==1])
                
                good_new = np.array(good_new)
                good_old = np.array(good_old)
                p0s[n] = good_new.reshape(-1, 1, 2)
                
                # Step 6-2: Find the Flow Lines
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    if a==c and b==d:
                        continue
                    new_line = FlowLine([c, d], [a, b], color[i])
                    
                    # Step 6-3: Ensure the Quality of Flow Lines (direction, length)
                    if new_line.angle > 180 and new_line.length() > MIN_FL_LEN:  
                        avg_len[n] = (avg_len[n] + new_line.length()*FL_UPDATE_RATE)/(1+FL_UPDATE_RATE)
                        if new_line.length() > avg_len[n]:
                            flow_lines.append(new_line)
                            cur_lines.append(new_line)
                            draw_mask = cv.line(draw_mask, (int(a), int(b)), 
                                    (int(c), int(d)), color[i].tolist(), 2)
                    frame = cv.circle(frame, (int(a), int(b)), 4, color[i].tolist(), -1)
                total_lines.extend(cur_lines)
            
            # Step 7: Find the cross points (CP) from each pair of flow lines
            for (l1, l2) in itertools.combinations(total_lines, 2):
                angle_diff = abs(l1.angle - l2.angle)

                # Step 7-1: Ensure that the CPs are generated 
                #           by 2 lines with enough angle difference
                if angle_diff < MIN_ANG_DIF or angle_diff > 360-MIN_ANG_DIF:
                    continue
                if abs(l1.start[0] - l2.start[0]) < WID*0.05:
                    continue
                x, y = cross_point(np.concatenate([l2.start, l2.stop]), 
                                    np.concatenate([l1.start, l1.stop]))

                # Step 7-2: Skip when there is no CP or the CP is lower than the flow lines
                if x is nan or y is nan or y > l1.start[1] or y > l2.start[1]:
                    continue
                if not vp.is_init() or vp.check_valid(x, y):

                    # Step 7-3: Store the valid CPs
                    new_cp = Point(isVP=False, x=x, y=y)
                    recent_cps.append(new_cp)
                    all_cps.append(new_cp)

                    # Step 7-4: Make Use of Stored CPs

                    # Step 7-4-1: If VP exists, update the current VP
                    if vp.is_init():
                        # culculate update direction
                        sum = np.array([0., 0.])
                        dif = []
                        for cp in recent_cps[-VP_REF_NUM:]:
                            dif.append(cp - vp)

                        mean = np.mean(dif, axis=0)
                        std = np.std(dif, axis=0)
                        c = 0
                        for d in dif:
                            # ignore too far points
                            if (np.less_equal(d, mean+std*MAX_CP_STD).all() and 
                                np.greater_equal(d, mean-std*MAX_CP_STD).all()):
                                sum = sum + d
                                c = c+1
                                
                        # update the VP with movement
                        if c != 0:
                            sum = sum/c
                            vp.update(sum[0], sum[1])
                            vp_history_xy.append((vp.x, vp.y))
                            all_vp.append(deepcopy(vp))
                            vp_ult = 0

                    # Step 7-4-2: If VP does not exist, initialize VP with enough CP references
                    elif (len(recent_cps) >= VP_REF_NUM):
                        sum = np.array([0., 0.])
                        for vp in recent_cps:
                            sum = sum + np.array([vp.x, vp.y])
                        sum = sum/VP_REF_NUM
                        vp.set(sum[0], sum[1])
                        vp_ult = 0

        # Step 8: Show the VP (Green Point)
        if vp.is_init():

            # Step 8-1: Hide VP when it has not updated in a long time (should be re-initialized)
            if vp_ult > HIDE_VP_THOLD :
                vp = Point(isVP=True)
                recent_cps = []
                avg_len = [MIN_FL_LEN, MIN_FL_LEN]
                print("vp hide")

            # Step 8-2: Draw on the frame and plot the data otherwise
            else:
                vp_history_xy.append((vp.x, vp.y))
                all_vp.append(deepcopy(vp))
                vl.update(all_vp[-VP_REF:], vp)
                if SHOW_VL > 0:
                    lp, rp, up, dp, st = vl()
                    if st:
                        frame = cv.line(frame, (int(lp[0]), int(lp[1])), (int(rp[0]), int(rp[1])), [0, 200, 50], 2)
                        frame = cv.line(frame, (int(up[0]), int(up[1])), (int(dp[0]), int(dp[1])), [0, 200, 50], 2)
                frame = cv.circle(frame, (int(vp.x), int(vp.y)), 6, [0, 255, 100], -1)                
                if SHOW_DNMC_PLOT:
                    plot_vp(all_vp, all_cps, vp, vl)
        
        # if you want to save the video with VP & flow lines
        if WRITE_VIDEO:
            out.write(cv.add(frame, draw_mask))

        # calculate FPS & draw on the frame
        new_time = time.time()
        fps = int(1/(new_time-prev_time))
        prev_time = new_time
        cv.putText(frame, "fps:"+str(fps), (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 4)
        cv.putText(frame, "fps:"+str(fps), (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        frame = cv.polylines(frame, [np.array([mask_points[1], mask_points[3], 
                                                mask_points[5], mask_points[7]])], 
                                                True, (0, 0, 100), 2)
        frame = cv.circle(frame, center, 6, [0, 0, 255], -1)
        img = cv.add(frame, draw_mask)
        cv.imshow('frame', img) # frame with VP & flow lines
        # cv.imshow('frame', frame) # frame with VP only
        # cv.imshow('processed frame', processed_frame)
        processed_old_frame = processed_frame.copy()
        
        # for key pressed down
        k = cv.waitKey(10) & 0xff
        if k == 27:
            cv.destroyAllWindows()
            if SHOW_DNMC_PLOT:
                plt.close()
            break
        elif k == 32:
            while (cv.waitKey(0) & 0xff != 32):
                continue
        elif k == 8:
            draw_mask = np.zeros_like(old_frame)

        
        # Step 9: If there are not enough tracking points, REPlace current points by the new points
        #                                              (or EXTend, based on your hyper-parameters)
        if (len(p0s[0]) + len(p0s[1])) < TP_NUM*TP_UPDATE_RATE or tp_ult == TP_UPDATE_TIME:
            tp_ult = 0
            new = []

            for j in range(2):
                new_ = []
                for i in range(2):
                    result = cv.goodFeaturesToTrack(processed_old_frame, mask = small_mask[j*2+i], **feature_params)
                    if result is not None:
                        new_.extend(result)
                new_ = np.array(new_, dtype=np.float32)
                new.append(new_)
            
            if len(new[0]) != 0 and len(new[1]) != 0:
                if FL_UPD_METH == "REP":
                    p0s = new
                elif FL_UPD_METH == "EXT":
                    for i in range(2):
                        p0s[i].extend(new[i])

        tp_ult += 1
        vp_ult += 1
    
    # end of while loop
    if SHOW_DNMC_PLOT:
        plt.ioff()
        plt.show()
        plt.close()

    print('-'*30)
    print("total lines: ", len(flow_lines))

    cv.destroyAllWindows()
    save_csv(vp_history_xy, video_name)
    cap.release()
    if WRITE_VIDEO:
        out.release()
# end of Run()

def data_statistic():
    x, y = read_csv(video_name)
    plt.figure(figsize=(12, 8), dpi=80)
    plt.title("VP distribution")
    plt.xlim(0, WID)
    plt.ylim(0, Hei)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x, y, 10)
    plt.gca().invert_yaxis()
    plt.axis('scaled')
    plt.show()

# customized pause function to focus on the OpenCV window
def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return

def plot_vp(vps, cps, best_vp, vl):
    """Plot the VP, vanishing line, VP history and recent cross points."""
    x = [row.x for row in vps[-VP_REF:]]
    y = [row.y for row in vps[-VP_REF:]]

    x_c = [row.x for row in cps[-VP_REF:]]
    y_c = [row.y for row in cps[-VP_REF:]]

    # plt.cla()
    plt.gca().clear()
    plt.gca().invert_yaxis()
    plt.xlim(WID//3*1, WID//3*2)
    plt.ylim(Hei//3, Hei//4*3)
    plt.scatter(WID/2, Hei/2, 100, 'r')
    plt.scatter(x_c, y_c, 10, 'y')
    plt.scatter(x, y, 20, 'b')
    plt.scatter(best_vp.x, best_vp.y, 100, 'g')
    if SHOW_VL > 1:
        lp, rp, up, dp, st = vl(mode='other')
        if st:
            plt.plot([lp[0], rp[0]], [lp[1], rp[1]])
            plt.plot([up[0], dp[0]], [up[1], dp[1]])
    plt.legend(["center", "cross points", "VPs history", "VP", "vanishing line"])
    mypause(0.01)

if __name__ == '__main__':
    setup()
    Run()
    # data_statistic()