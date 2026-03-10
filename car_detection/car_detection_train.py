from roboflow import Roboflow
from ultralytics import YOLO

def car_v11():
  
    print("데이터셋 다운로드...")
    # rf = Roboflow(api_key="RH2HqpxJIMC9ESalK3mG")
    # project = rf.workspace("cjy-ruezi").project("vehicle-hdw2i-z8a8u")
    # version = project.version(1)
    # dataset = version.download("yolov11")
    rf = Roboflow(api_key="RH2HqpxJIMC9ESalK3mG")
    project = rf.workspace("cjy-ruezi").project("vehicle-j080n-wbse5")
    version = project.version(2)
    dataset = version.download("yolov11")
                
                
    print(f"다운로드 완료")

 
    print("YOLO11 모델 학습 시작...")
    model = YOLO("yolo11n.pt") 
    model.train(
        data=f"{dataset.location}/data.yaml",  
        device=0,
        epochs=100,                           
        patience=20,                    
        imgsz=640,                           
        batch=8,
        name='YOLOv11n_car_detection2',
        exist_ok=True,  # 기존 폴더 덮어쓰기
        cache=False     # 중요: 이전 데이터 캐시 무시하고 새로 읽기
    )
    print("학습 완료")

if __name__ == "__main__":
    car_v11()