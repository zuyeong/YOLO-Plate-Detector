import cv2
from ultralytics import YOLO

# 모델 로드
model = YOLO("/home/cjy/workspace/original/YOLOv11n.pt")

results = model.track(
    source="/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/original_video.mp4", 
    save=True,
    tracker="/home/cjy/workspace/video/bytetrack.yaml", 
    conf=0.25,     
    persist=True,
    device='0',
    stream=True 
)

for r in results:
    pass  # save=True