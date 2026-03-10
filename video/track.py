from ultralytics import YOLO

model = YOLO("/home/cjy/workspace/original/YOLOv11n.pt")

results = model.track(
    source="/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/original_video.mp4", 
    save=True,
    tracker="bytetrack.yaml"
)