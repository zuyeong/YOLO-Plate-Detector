import cv2
import os
import shutil  # 파일 복사를 위해 필요
from ultralytics import YOLO
from tqdm import tqdm

MODEL = 'runs/detect/YOLOv11n_car_detection/weights/best.pt'
SAVE_DIR = '/mnt/hdd_6tb/cjy/processed_full' 

DATA_DIR = [
    {
        "split": "train",
        "image_dir": "/home/cjy/workspace/original/processed_original/images/train",
        "label_dir": "/home/cjy/workspace/original/processed_original/labels/train"
    },
    {
        "split": "validation", 
        "image_dir": "/home/cjy/workspace/original/processed_original/images/validation",
        "label_dir": "/home/cjy/workspace/original/processed_original/labels/validation"
    }
]

ID_PLATE = 0    
ID_CAR = 1      

def car_coordinates():
    model = YOLO(MODEL)

    for config in DATA_DIR:
        split_name = config["split"]
        img_dir = config["image_dir"]
        origin_label_dir = config["label_dir"]

        save_img_dir = os.path.join(SAVE_DIR, 'images', split_name)
        save_lbl_dir = os.path.join(SAVE_DIR, 'labels', split_name)
        
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_lbl_dir, exist_ok=True)

        print(f"[{split_name}] 작업 시작...")
            
        img_list = [f for f in os.listdir(img_dir) if f.endswith(('.png'))]

        for file_name in tqdm(img_list, desc=f"{split_name} 처리"):
            # 경로 설정
            img_path = os.path.join(img_dir, file_name)
            txt_name = os.path.splitext(file_name)[0] + ".txt"
            
            origin_txt_path = os.path.join(origin_label_dir, txt_name)
            save_txt_path = os.path.join(save_lbl_dir, txt_name)     # 라벨 저장 경로
            save_img_path = os.path.join(save_img_dir, file_name)    # 이미지 저장 경로

            # 이미지 읽기
            img = cv2.imread(img_path)
            if img is None: continue
            h_img, w_img, _ = img.shape

            # 기존 번호판 라벨 읽기
            plates = []
            if os.path.exists(origin_txt_path):
                with open(origin_txt_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = list(map(float, line.strip().split())) # 리스트에 있는 문자를 실수로 변환 
                        plates.append(parts[1:])
            if not plates: continue 

            # 차량 탐지
            results = model(img, verbose=False, conf=0.5)
            final_labels = [f"{ID_PLATE} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {p[3]:.6f}" for p in plates]

            for box in results[0].boxes:
                cx, cy, cw, ch = box.xywhn[0].tolist()         
                c_x1, c_y1, c_x2, c_y2 = box.xyxyn[0].tolist() 

                has_plate = False
                for p in plates:
                    p_cx, p_cy, _, _ = p   # 번호판 중심 좌표
                    if (c_x1 < p_cx < c_x2) and (c_y1 < p_cy < c_y2): # 차량 좌표 안에 번호판이 있는지 판단 
                        has_plate = True
                        break 
    
                if has_plate:
                    final_labels.append(f"{ID_CAR} {cx:.6f} {cy:.6f} {cw:.6f} {ch:.6f}") 

            # 라벨 파일 저장
            with open(save_txt_path, 'w') as f:
                f.write('\n'.join(final_labels))
            
            # 이미지 파일 복사
            shutil.copy(img_path, save_img_path)

    print("작업 완료")

if __name__ == "__main__":
    car_coordinates()