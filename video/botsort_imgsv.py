import cv2
from ultralytics import YOLO

VIDEO_PATH = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/original_video.mp4"
OUTPUT_PATH = "/home/cjy/workspace/video/inference/botsort_imgsv.mp4"
MODEL_PLATE_PATH = "/home/cjy/workspace/original/YOLOv11n.pt" 

model = YOLO(MODEL_PLATE_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

print("고해상도 ID 추적")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame, 
        conf=0.25,
        imgsz=1920,       
        persist=True,     
        tracker="botsort.yaml", 
        verbose=False, 
        device='0' 
    )
    
    # 알아서 박스와 ID를 다 그려줌
    annotated_frame = results[0].plot()

    out.write(annotated_frame)

cap.release()
out.release()
print("완료!")