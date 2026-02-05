from roboflow import Roboflow
from ultralytics import YOLO

def car_v11():
  
    print("데이터셋 다운로드...")
    rf = Roboflow(api_key="RH2HqpxJIMC9ESalK3mG")
    project = rf.workspace("cjy-ruezi").project("vehicle-hdw2i-z8a8u")
    version = project.version(1)
    dataset = version.download("yolov11")
    print(f"다운로드 완료")

 
    print("YOLO11 모델 학습 시작...")
    model = YOLO("yolo11n.pt") 
    model.train(
        data=f"{dataset.location}/data.yaml",  
        device=0,
        epochs=100,                           
        patience=20,            # 20번 동안 성능 향상 없으면 조기 종료          
        imgsz=640,                           
        batch=8,
        name='YOLOv11n_car_detection'         
    )
    print("학습 완료")

if __name__ == "__main__":
    car_v11()