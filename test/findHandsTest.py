import cv2
import sys

from numpy import true_divide
sys.path.append('/'.join(sys.path[0].split('/')[:6]))
print(sys.path)
from findHands import handDetector
import time
from scipy.spatial import distance

def check_landmarks(landmarks):
    x1, y1, = landmarks[4][1], landmarks[4][2]
    x2, y2 = landmarks[8][1], landmarks[8][2]
    dist = distance.euclidean((x1,y1), (x2,y2))
    print(dist)
    if dist < 40:
        return True
    return False

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = handDetector()
while True:
    success, img = cap.read()
    img = detector.findHandsInImage(img)
    landmark_list = detector.findPosition(img)
    if len(landmark_list) != 0:
        if(check_landmarks(landmark_list)):
            cv2.putText(img, "Pointer and thumb are close together", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        else:
            cv2.putText(img, "Pointer and thumb are NOT close together", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)