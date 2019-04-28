import cv2
import numpy as np
from collections import deque

glhsv = np.array([65, 60, 60])
guhsv = np.array([80, 255, 255])

blhsv = np.array([95, 150, 100])
buhsv = np.array([180,255,255])

rlhsv=np.array([136,87,111])
ruhsv=np.array([180,255,255])

kernel = np.ones((5, 5), np.uint8)
lower = {0:(166, 84, 141), 1:(65, 60, 60), 2:(97, 100, 117)}
upper = {0:(186,255,255), 1:(80, 255, 255), 2:(117,255,255)}
colors = {0:(0,0,255), 1:(0,255,0), 2:(255,0,0)}
bpoints = [deque(maxlen=512)]   
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]
bindex = 0
gindex = 0
vindex = 0
colorIndex = 0
paintWindow = np.zeros((471,636,3)) + 255
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)
camera = cv2.VideoCapture(0)

while True:

    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 for key, value in upper.items():

        kernel = np.ones((9,9),np.uint8)
        mask = cv2.inRange(hsv, lower[key], upper[key])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        if len(cnts) > 0:

            cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
            points = [bpoints, gpoints, rpoints]
            for i in range(len(points)):
                for j in range(len(points[i])):
                    for k in range(1, len(points[i][j])):
                        if points[i][j][k - 1] is None or points[i][j][k] is None:
                            continue
                        cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[key], 3)
                        cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[key], 4)

            M = cv2.moments(cnt)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            bpoints[bindex].appendleft(center)

    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('w'):
        cv2.imwrite(str("1")+'.jpg',paintWindow)
        break
camera.release()
cv2.destroyAllWindows()
