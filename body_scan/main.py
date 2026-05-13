import cv2
import cvzone
from ultralytics import YOLO
import numpy as np

cap = cv2.VideoCapture("./NewVideo.mov")
model = YOLO("yolov8n-pose.pt")

while True:
  ret, frame = cap.read()
  
  if not ret:
    cap = cv2.VideoCapture("./Video.mov")
    continue
    
  frame = cv2.resize(frame, (640,720))
  width,height = frame.shape[:2]
  blank_image = np.zeros((width, height, 3), dtype=np.uint8)
  
  results = model(frame)
  frame = results[0].plot()
  
  for keypoints in results[0].keypoints.data:
    keypoints = keypoints.cpu().numpy()
    
    for i, keypoint in enumerate(keypoints):
      x, y, confidence = keypoint
      
      if confidence > 0.7:
        cv2.circle(blank_image, (int(x), int(y)), 5, (0,255,0), cv2.FILLED)
        cv2.putText(blank_image, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        
        
  connections = [
    (3,1), (1,0), (0,2), (2,4), (1,2), (4,6), (3,5), (5,6), (5,7), (7,9),
    (6,8), (8,10), (11,12), (11,13), (13,15), (12,14),
    (14,16), (5,11), (6,12)
  ]
  
  for part_a, part_b in connections:
    x1,y1,confidence1 = keypoints[part_a]
    x2,y2,confidence2 = keypoints[part_b]
  
    if confidence1 > 0.7 and confidence2 > 0.7:
      cv2.line(blank_image, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)

  output = cvzone.stackImages([frame, blank_image], 2, 1)
  cv2.imshow("frame", output)
  # cv2.imshow("frames", blank_image)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  
  
cap.release()
cv2.destroyAllWindows()