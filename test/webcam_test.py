import cv2
import time 
cap = cv2.VideoCapture(0)
pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cTime = time.time()
    frames_per_second = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(frames_per_second)}', (20, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2) 
    cv2.imshow("Test", img)
    cv2.waitKey(1)