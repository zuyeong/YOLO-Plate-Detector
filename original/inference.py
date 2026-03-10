# from ultralytics import YOLO
# import glob
# import os

# model = YOLO("YOLOv11n.pt")
# test_image_dir = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/Test/01.원천데이터/**/*.jpg"
# # image_file = glob.glob(os.path.join(test_image_dir, '**', '*.jpg'), recursive=True)
# # total_images = len(image_file)
# # print(f"검색된 총 이미지 개수: {total_images}장")

# model.predict(
#     source=test_image_dir,     # test 이미지 경로
#     save=True,                 # 결과 이미지 저장
#     project="runs/detect",     # 저장될 대분류 폴더
#     name="inference_results",  # 저장될 소분류 폴더
#     exist_ok=True,             # 덮어쓰기 허용
#     conf=0.05,                 # 25% 이상 확신하는 것만 박스 그리기
#     device='3',
#     line_width=3,     
#     show_conf=True,  
#     show_labels=False
# )

import cv2
import glob
import os
from ultralytics import YOLO

MODEL_PATH = "YOLOv11n.pt"
SOURCE_PATTERN = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/Test/01.원천데이터/**/*.jpg"
SAVE_DIR = "runs/detect/inference_results"

BOX_COLOR = (255, 0, 0)      
TEXT_COLOR = (255, 255, 255)      
FONT_SCALE = 1.0            
THICKNESS = 2


model = YOLO(MODEL_PATH)
img_list = glob.glob(SOURCE_PATTERN, recursive=True)

for i, img_path in enumerate(img_list):
    if (i + 1) % 100 == 0:
        print(f"Processing [{i+1}/{len(img_list)}]")

    try:
        img = cv2.imread(img_path)
        if img is None: continue

        results = model(img, conf=0.05, device='3', verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 좌표 및 정보 추출
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                
                # 라벨 텍스트: "클래스명 0.85"
                label_text = f"{model.names[cls]} {conf:.2f}"

                #  박스 그리기
                cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 3)
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS)
                
                if y1 - h - 10 < 0:
                    cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h + 10), BOX_COLOR, -1)
                    text_y = y1 + h + 5
                else:
                    cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), BOX_COLOR, -1)
                    text_y = y1 - 5

                cv2.putText(img, label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, THICKNESS)


        file_name = os.path.basename(img_path)
        save_path = os.path.join(SAVE_DIR, file_name)
        cv2.imwrite(save_path, img)

    except Exception as e:
        print(f"Error: {e}")
        continue

print(f"완료")