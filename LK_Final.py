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
import matplotlib.pyplot as plt
import pickle
from scipy import stats

WID = 860               
"window width"
Hei = 540               
"window height"
TP_NUM = 20             
"maximum tracking points at the same time"
VP_NUM = 15
"number of recent cross points references to update the best VP per round"
VP_UPDATE_RATE = 0.3
"update rate to the best VP"
FL_UPDATE_RATE = 0.01
"update rate for calculating average length of flow lines"
TP_UPDATE_RATE = 0.3
"update rate tracking points when they are not enough"
TP_UPDATE_TIME = 10
"number of frames to update tracking points "
MIN_ANG_DIF = 25
"(degree) max acceptable angle difference of 2 lines to construct cross pt"
MAX_CP_STD = 1.0
"max acceptable standard deviation range of new cross points (distance between cross point and best VP)"
MIN_FL_LEN = 2.
"shortest acceptable length of flow lines"
CP_THOLD = 1/15
"max distance acceptable from new cross point to the VP (proportion of the window)"
HIDE_VP_THOLD = 50
"the number of frames that VP has not updated to reset VP"
FL_UPDATE_METH = "EXT"
"EXTend flow points or REPlace them by new points"


def setup():
    """Setup global variables"""
    global cap, feature_params, lk_params, color, video_name

    plt.figure(figsize=(12, 8), dpi=80)
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

class VP:
    """
    Vanishing Point/Cross Point Class
    =====
    A class to represent a vanishing point or cross point.

    ...

    Attributes
    ----------
    x : float
        x coordinate value
    y : float
        y coordinate value

    Methods
    -------
    is_init():
        Check whether the point is initialized.

    has_moved():
        Check whether the point has moved.

    set(x, y):
        Initialize the point with coordinate values.

    update(x, y):
        Update the point with movement coordinate values.

    check_valid(best):
        Check whether the new cross point is too far from the current vanishing point
    """
    def __init__(self, isbest: bool, x=None, y=None) -> None:
        self._isinit = not isbest
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
        if (not self._isinit):
            raise Exception("VP is not initialized")
        self._moved = True
        self.x = self.x + x * VP_UPDATE_RATE
        self.y = self.y + y * VP_UPDATE_RATE

    def is_init(self) -> bool:
        return self._isinit

    def has_moved(self) -> bool:
        return self._moved

    def check_valid(self, best) -> bool:
        """Check whether the new cross point is too far from the current vanishing point"""
        return (np.absolute(self-best) < np.array([WID*CP_THOLD, Hei*CP_THOLD])).all()

class VL:
    """
    Vanishing Line Class
    =====
    A class to represent a pair of vanishing lines.

    ...

    Attributes
    ----------
    None

    Methods
    -------
    update(vps: list, best_vp: VP):
        Update the line pairs with a list of VP history and the current best vanishing point.
    """
    def __init__(self):
        self._isinit = False

    def __call__(self, mode = 'best_point'):
        if not self._isinit:
            return None, None, None, None
        if mode == 'best_point':
            return self._lp, self._rp, self._up, self._dp
        else :
            return self._calculate_endpt()

    def update(self, vps: list, best_vp: VP):
        """Update the line pairs with a list of VP history and the current best vanishing point."""
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
        return l, r, u, d

class FlowLine:
    """
    Optical Flow Line Segment Class
    =====
    A class to represent a optical flow line.

    ...

    Attributes
    ----------
    start : array
        Start point.
    stop : array
        End point.
    length : float
        Length of the line.
    angle : float
        Angle on the xy-coordinate system of the line.
    color : array
        BRG value of the line color.

    Methods
    -------
    get_info(vps: list, best_vp: VP):
        Print the information of the line.
    """
    def __init__(self, start=[0, 0], stop=[0, 0], color=[0,0,0]):
        self.start = np.array(start) 
        self.stop = np.array(stop)
        self._vector = np.subtract(np.multiply(self.stop, [1, -1]), 
                                np.multiply(self.start, [1, -1]))
        self._length = np.round(np.linalg.norm(self._vector), 2)
        self.angle = angle_between(self._vector, [1, 0])
        self.color = color

    def __len__(self):
        return self._length

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

def checkInside(pt, mask = [], st = []):
    """
        Brief
        ---
        Check whether the tracking point is still inside the ROI.

        Parameters
        ---
        mask : array
            A list coordinates of the boundary points.
        st : array
            A list of current status.

        Returns
        ---
        isInside : boolean
    """
    status = []
    for id in range(len(pt)):
        if st[id,0] == 0 or floor(pt[id,0,1])>mask.shape[0] or floor(pt[id,0,0])>mask.shape[1] :
            status.append([0])
        else:
            status.append([mask[floor(pt[id,0,1]), floor(pt[id,0,0])] > 0])
    return np.array(status)

def modify_contrast_and_brightness2(img, brightness=0 , contrast=100):
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
    processed_img = modify_contrast_and_brightness2(processed_img)
    # test = np.clip(test, 0, 255)
    # test = np.uint8(test)
    # test[mask == 0] = 0
    # test = np.bitwise_and(test, mask)
    # avg_shi = np.round(np.mean(test[mask]), 2)
    # avg_shi = np.round(np.mean(test[border:]), 2)
    # avg_shi = min(max(50, avg_shi), 100)
    processed_img = cv.GaussianBlur(processed_img, (5, 5), 0)
    # processed_frame = cv.GaussianBlur(processed_frame, (3, 3), 0)
    # test = cv.threshold(test, np.mean(test[test!=0]) + np.std(test[test!=0])*1.5, 255, cv.THRESH_BINARY)[1]
    # kernel = np.ones((3,3), np.uint8)
    # test = cv.erode(test, None, iterations=1)
    # test = cv.dilate(test, None, iterations=1)
    return processed_img
    
def Run():
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    ratio = cap.get(cv.CAP_PROP_FRAME_HEIGHT)/cap.get(cv.CAP_PROP_FRAME_WIDTH)
    global Hei
    Hei = int(WID*ratio)
    old_frame = imutils.resize(old_frame, width=int(WID))     
    center = (int(WID/2), int(Hei/2))
    
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

    mask_points = np.array([[ WID//2,                                   (bounds["outerD"]+bounds["innerU"])//2],    #中間 [0]
                            [ bounds["outerL"],                          bounds["outerD"]],                         #左下 [1]
                            [ WID//2,                                    bounds["outerD"]],                         #中下 [2]
                            [ bounds["outerR"],                          bounds["outerD"]],                         #右下 [3]
                            [(bounds["outerR"]+bounds["innerR"])//2,    (bounds["outerD"]+bounds["innerU"])//2],    #右中 [4]
                            [ bounds["innerR"],                          bounds["innerU"]],                         #右上 [5]
                            [ WID//2,                                    bounds["innerU"]],                         #中上 [6]
                            [ bounds["innerL"],                          bounds["innerU"]],                         #左上 [7]
                            [(bounds["outerL"]+bounds["innerL"])//2,    (bounds["outerD"]+bounds["innerU"])//2]])   #左中 [8]
    
    # cut a mask
    mask = np.zeros(old_frame.shape[:2], dtype=np.uint8)
    mask = cv.fillPoly(mask, [np.array([mask_points[1], mask_points[3], mask_points[5], mask_points[7]])], 255)
                                        
    cv.imshow('frame', mask)
    cv.waitKey(200)

    small_mask = []
    for i in range(4):
        small_mask.append(np.zeros_like(mask))

    small_mask[0] = cv.fillPoly(small_mask[0], [np.array([mask_points[0], mask_points[8], mask_points[1], mask_points[2]])], 255)
    small_mask[1] = cv.fillPoly(small_mask[1], [np.array([mask_points[0], mask_points[2], mask_points[3], mask_points[4]])], 255)
    small_mask[2] = cv.fillPoly(small_mask[2], [np.array([mask_points[0], mask_points[4], mask_points[5], mask_points[6]])], 255)
    small_mask[3] = cv.fillPoly(small_mask[3], [np.array([mask_points[0], mask_points[6], mask_points[7], mask_points[8]])], 255)

    for i in range(4):
        cv.imshow('frame', small_mask[i])
        cv.waitKey(100)

    processed_old_frame = process_img(old_frame)

    p0 = []
    for i in range(4):
        new = cv.goodFeaturesToTrack(processed_old_frame, mask = small_mask[i], **feature_params)
        if new is not None:
            p0.extend(new)
    p0 = np.array(p0)
    
    vp_history_xy, all_vp = [], []
    recent_cps, all_cps = [], []
    flow_lines = []
    vl = VL()
    best_vp = VP(isbest=True)
    prev_time = time.time()
    avg_len = MIN_FL_LEN

    vp_ult = 0  #: number of frames passed since last time vanishing point updated
    tp_ult = 0  #: number of frames passed since last time tracking point updated
    all_lines_frame = np.zeros_like(old_frame, dtype=np.uint8)
    
    while(1):
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            cv.destroyAllWindows()
            # exit()
            break
        
        frame = imutils.resize(frame, width=int(WID))
        processed_frame = process_img(frame)

        """
        # Bird's eye view
        src = np.float32([[0, Hei*0.85], [WID, Hei*0.85], [WID//3, Hei*0.6], [WID//3*2, Hei*0.6]])
        dst = np.float32([[WID*0.4, Hei], [WID*0.6, Hei], [0, 0], [WID, 0]])
        Mx = cv.getPerspectiveTransform(src, dst) # The transformation matrix
        Minv = cv.getPerspectiveTransform(dst, src) # Inverse transformation
        warped_img = cv.warpPerspective(frame, Mx, (WID, Hei))
        cv.imshow('Bird\'s Eye View', warped_img)

        cur_frame = np.zeros_like(processed_frame)
        """
        good_new = []
        good_old = []
        cur_lines = []
        
        if len(p0) != 0:
            # calculate optical flow
            p1, st, err = cv.calcOpticalFlowPyrLK(processed_old_frame, 
                            processed_frame, p0, None, **lk_params)
            
            # Select good points
            # Check if the line is still inside the mask
            if p1 is not None:
                filter = checkInside(p1, mask, st)
                st[~filter] = 0
                good_new.extend(p1[st==1])
                good_old.extend(p0[st==1])
            
            good_new = np.array(good_new)
            good_old = np.array(good_old)
            p0 = good_new.reshape(-1, 1, 2)

            # Find The Flow Lines
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                if a==c and b==d:
                    continue
                new_line = FlowLine([c, d], [a, b], color[i])
            
                if (new_line.angle > 180 and len(new_line) > MIN_FL_LEN):  
                    if (len(new_line) > avg_len):
                        flow_lines.append(new_line)
                        cur_lines.append(new_line)
                        all_lines_frame = cv.line(all_lines_frame, (floor(a), floor(b)), 
                                        (floor(c), floor(d)), new_line.color.tolist(), 2)
                    avg_len = (avg_len+len(new_line)*FL_UPDATE_RATE)/(1+FL_UPDATE_RATE)
                frame = cv.circle(frame, (int(a), int(b)), 4, color[i].tolist(), -1)

            # Find the cross points from each pair of flow lines
            for pair in itertools.combinations(cur_lines, 2):
                l1, l2 = pair
                angle_diff = abs(l1.angle - l2.angle)
                if (angle_diff > MIN_ANG_DIF and angle_diff < 360-MIN_ANG_DIF):
                    x, y = cross_point(np.concatenate([l2.start, l2.stop]), 
                                        np.concatenate([l1.start, l1.stop]))
                    if x is nan or y is nan:
                        continue
                    if y > l1.start[1] or y > l2.start[1]:
                        # if the position of the cross point is lower than the flow line
                        continue

                    new_cp = VP(False, x, y)
                    recent_cps.append(new_cp)
                    all_cps.append(new_cp)

                    # if the best vanishing point is found
                    if (best_vp.is_init()):
                        if not new_cp.check_valid(best_vp):
                            recent_cps.pop()
                            all_cps.pop()
                            del new_cp
                            continue
                        sum = np.array([0., 0.])
                        dif = []
                        for vp in recent_cps[-VP_NUM:]:
                            dif.append(vp - best_vp)

                        mean = np.mean(dif, axis=0)
                        std = np.std(dif, axis=0)
                        c = 0
                        for d in dif:
                            if (np.less_equal(d, mean+std*MAX_CP_STD).all() and 
                                np.greater_equal(d, mean-std*MAX_CP_STD).all()):
                                sum = sum + d
                                c = c+1
                                
                        # update the best VP with movement
                        if (c != 0):
                            sum = sum/c
                            best_vp.update(sum[0], sum[1])
                            vp_ult = 0

                    # initialize VP
                    elif (len(recent_cps) >= VP_NUM):
                        sum = np.array([0., 0.])
                        for vp in recent_cps:
                            sum = sum + np.array([vp.x, vp.y])
                        sum = sum/VP_NUM
                        best_vp.set(sum[0], sum[1])
                        vp_ult = 0
            
            # VP updated long time ago
            if best_vp.is_init() and vp_ult > HIDE_VP_THOLD :
                best_vp = VP(True)
                recent_cps = []
                print("hide")

        # show the best VP (Green)
        if (best_vp.is_init()):
            vp_history_xy.append((best_vp.x, best_vp.y))
            all_vp.append(deepcopy(best_vp))
            vl.update(all_vp, best_vp)
            lp, rp, up, dp = vl()
            if (lp is not None):
                frame = cv.line(frame, (int(lp[0]), int(lp[1])), (int(rp[0]), int(rp[1])), [0, 200, 50], 2)
                frame = cv.line(frame, (int(up[0]), int(up[1])), (int(dp[0]), int(dp[1])), [0, 200, 50], 2)
            frame = cv.circle(frame, (int(best_vp.x), int(best_vp.y)), 6, [0, 255, 100], -1)
            all_lines_frame = cv.circle(all_lines_frame, 
                                (int(best_vp.x), int(best_vp.y)), 2, [0, 255, 100], -1)
            
            plot_vp(all_vp, all_cps, best_vp, vl, 0)
        
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
        
        cv.imshow('frame', frame)
        cv.imshow('processed frame', processed_frame)
        
        k = cv.waitKey(10) & 0xff
        if k == 27:
            cv.destroyAllWindows()
            plt.close()
            break
        elif k == 32:
            while (cv.waitKey(0) & 0xff != 32):
                continue

        # update the previous frame and previous points
        processed_old_frame = processed_frame.copy()
        
        # when # tracking points is not enough
        if len(p0) < TP_NUM*TP_UPDATE_RATE or tp_ult == TP_UPDATE_TIME:
            tp_ult = 0
            new = []
            for i in range(4):
                result = cv.goodFeaturesToTrack(processed_old_frame, 
                            mask = small_mask[i], **feature_params)
                if result is not None:
                    new.extend(result)
            new = np.array(new, dtype=np.float32)
            
            # EXTend p0 or REPlace p0 by new points
            if new is not None:
                if FL_UPDATE_METH == "REP":
                    p0 = new.reshape(-1, 1, 2)
                    
                elif FL_UPDATE_METH == "EXT":
                    p0 = np.append(p0, new).reshape(-1, 1, 2).astype(np.float32)
                    if (len(p0) > TP_NUM) :
                        p0 = p0[-TP_NUM:]

        tp_ult += 1
        vp_ult += 1
    
    plt.ioff()
    plt.show()

    print('-'*30)
    print("total lines: ", len(flow_lines))
    all_lines_frame = cv.circle(all_lines_frame, 
                        (int(WID/2), int(Hei/2)), 6, [0, 0, 255], -1)

    cv.imshow('all_lines', all_lines_frame)
    while cv.waitKey(0) != 27:
        continue

    cv.destroyAllWindows()
    save_csv(vp_history_xy, video_name)
    cap.release()
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

def plot_vp(vps, cps, best_vp, vl, num):
    x = [row.x for row in vps[-num:]]
    y = [row.y for row in vps[-num:]]

    x_c = [row.x for row in cps[-num:]]
    y_c = [row.y for row in cps[-num:]]

    plt.clf()
    plt.title(f"Recent {num} Points")
    # plt.figure(figsize=(12, 8), dpi=80)
    plt.xlim(WID//3*1, WID//3*2)
    plt.ylim(Hei//3, Hei//4*3)
    plt.gca().invert_yaxis()
    plt.axis('scaled')
    plt.ylabel("y axis")
    plt.xlabel("x axis")
    plt.scatter(WID/2, Hei/2, 100, 'r')
    plt.scatter(x_c, y_c, 10, 'y')
    plt.scatter(x, y, 20, 'b')
    plt.scatter(best_vp.x, best_vp.y, 100, 'black')
    lp, rp, up, dp = vl(mode='other')
    if lp is not None and up is not None:
        # print(lp, rp)
        plt.plot([lp[0], rp[0]], [lp[1], rp[1]])
        plt.plot([up[0], dp[0]], [up[1], dp[1]])
    plt.legend(["center", "cross points", "vps", "best point", "vanishing line"])
    plt.pause(0.001)

if __name__ == '__main__':
    setup()
    Run()
    # data_statistic()