import cv2
import cvzone
from ultralytics import YOLO
import numpy as np

cap = cv2.VideoCapture("./Video.mov")

while True:
  ret, frame = cap.read()
  
  cv2.imshow("frame", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  
  
cap.release()
cv2.destroyAllWindows()