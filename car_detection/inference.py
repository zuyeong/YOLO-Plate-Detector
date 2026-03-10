import cv2
import glob
import os
import torch
import torchvision
from ultralytics import YOLO

# ================= 설정 =================
IMG_DIR = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/Test/01.원천데이터/**/*.jpg"
OUTPUT_DIR = "/home/cjy/workspace/car_detection/model_A_inference"

MODEL_CAR_PATH = "/home/cjy/workspace/car_detection/runs/detect/YOLOv11n_car_detection/weights/best.pt"
MODEL_PLATE_PATH = "/home/cjy/workspace/car_detection/runs/detect/train2/weights/best.pt" 

MARGIN = 0.15      # 4분할 겹침 비율
CONF_CAR = 0.45    # 차량 신뢰도
CONF_PLATE = 0.35  # 번호판 신뢰도
IOU = 0.3          # 중복 제거 기준

# 차량을 자를 때 여유 공간
CROP_PADDING = 0.1 # 10% 여유

COLOR_CAR = (0, 255, 0)
COLOR_PLATE = (0, 0, 255)
# ========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 모델 3개 로드 
car_model = YOLO(MODEL_CAR_PATH)
plate_model = YOLO(MODEL_PLATE_PATH)

img_paths = glob.glob(IMG_DIR, recursive=True)
total_images = len(img_paths)

for k, img_path in enumerate(img_paths):
    if (k + 1) % 100 == 0:
        print(f"Processing [{k+1}/{total_images}] ...")
        
    img = cv2.imread(img_path)
    if img is None: continue
    
    h, w = img.shape[:2]
    h2, w2 = h // 2, w // 2
    m_h, m_w = int(h * MARGIN), int(w * MARGIN)

    # 4분할 영역
    crops_info = [
        (img[0:h2+m_h, 0:w2+m_w],      0,      0),      
        (img[0:h2+m_h, w2-m_w:w],      w2-m_w, 0),       
        (img[h2-m_h:h, 0:w2+m_w],      0,      h2-m_h), 
        (img[h2-m_h:h, w2-m_w:w],      w2-m_w, h2-m_h)  
    ]

    # 배치 추론 (차량 모델)
    batch_imgs = [c[0] for c in crops_info]
    results_car = car_model(batch_imgs, conf=CONF_CAR, verbose=False, device='0')

    # 좌표 통합
    car_boxes_list = []
    car_scores_list = []
    
    for i, res in enumerate(results_car):
        if len(res.boxes) == 0: continue
        
        # 좌표 보정
        boxes = res.boxes.xyxy.cpu()
        boxes[:, [0, 2]] += crops_info[i][1] 
        boxes[:, [1, 3]] += crops_info[i][2] 
        
        car_boxes_list.append(boxes)
        car_scores_list.append(res.boxes.conf.cpu())

    final_cars = []
    final_scores = []

    if len(car_boxes_list) > 0:
        all_boxes = torch.cat(car_boxes_list)
        all_scores = torch.cat(car_scores_list)
        
        # NMS
        keep_indices = torchvision.ops.nms(all_boxes, all_scores, IOU)
        
        final_cars = all_boxes[keep_indices]
        final_scores = all_scores[keep_indices]

    # ========================================
    # 번호판 탐지
    
    # 감지된 차량이 없으면 넘어감
    if len(final_cars) == 0:
        filename = os.path.basename(img_path)
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), img)
        continue

    for idx, car_box in enumerate(final_cars):
        cx1, cy1, cx2, cy2 = map(int, car_box)
        car_score = float(final_scores[idx])

        # Padding
        bw = cx2 - cx1
        bh = cy2 - cy1
        pad_w = int(bw * CROP_PADDING)
        pad_h = int(bh * CROP_PADDING)

        crop_x1 = max(0, cx1 - pad_w)
        crop_y1 = max(0, cy1 - pad_h)
        crop_x2 = min(w, cx2 + pad_w)
        crop_y2 = min(h, cy2 + pad_h)

        # 차량 이미지 Crop
        car_img_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if car_img_crop.size == 0: continue

        # 차량 박스 그리기
        cv2.rectangle(img, (cx1, cy1), (cx2, cy2), COLOR_CAR, 2)
        car_label = f"Car {car_score:.2f}"
        (tw, th), _ = cv2.getTextSize(car_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(img, (cx1, cy1 - 25), (cx1 + tw, cy1), COLOR_CAR, -1)
        cv2.putText(img, car_label, (cx1, cy1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # crop 이미지 넣어줌 
        results_plate = plate_model(car_img_crop, conf=CONF_PLATE, verbose=False, device='0')

        for res in results_plate:
            for box in res.boxes:
                # 번호판 좌표 (Crop 기준)
                px1, py1, px2, py2 = map(int, box.xyxy[0])
                p_conf = float(box.conf[0])

                # 좌표 변환: (Crop 내부) + (Crop 시작점 crop_x1, crop_y1)
                real_px1 = px1 + crop_x1
                real_py1 = py1 + crop_y1
                real_px2 = px2 + crop_x1
                real_py2 = py2 + crop_y1

                # 번호판 박스 그리기
                cv2.rectangle(img, (real_px1, real_py1), (real_px2, real_py2), COLOR_PLATE, 2)
                label = f"Plate {p_conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                # 텍스트 배경
                cv2.rectangle(img, (real_px1, real_py1 - 25), (real_px1 + tw, real_py1), COLOR_PLATE, -1)
                cv2.putText(img, label, (real_px1, real_py1 - 3), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    filename = os.path.basename(img_path)
    save_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(save_path, img)

print("완료")