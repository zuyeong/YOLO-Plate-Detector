import cv2
import glob
import os
import torch
import torchvision
import json
from ultralytics import YOLO

# ================= 설정 =================
IMG_DIR = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/Test/01.원천데이터/**/*.jpg"

MODEL_CAR_PATH = "/home/cjy/workspace/car_detection/runs/detect/YOLOv11n_car_detection/weights/best.pt"
MODEL_PLATE_PATH = "/home/cjy/workspace/car_detection/runs/detect/train2/weights/best.pt" 

MARGIN = 0.15      # 4분할 겹침 비율
CONF_CAR = 0.6    # 차량 신뢰도
CONF_PLATE = 0.35  # 번호판 신뢰도
IOU_CAR = 0.3      # 차량 중복 제거(NMS) 기준
CROP_PADDING = 0.1 # 차량 크롭 여유 공간 (10%)
IOU_EVAL = 0.5     # 정답으로 인정할 IoU 기준 (0.5 이상이면 정답)
DEVICE = '0'       # GPU 번호
# ========================================

# IoU 계산 함수
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

print("모델 로드 중...")
car_model = YOLO(MODEL_CAR_PATH)
plate_model = YOLO(MODEL_PLATE_PATH)

image_files = glob.glob(IMG_DIR, recursive=True)
total_images = len(image_files)

gt_total = 0      # Ground Truth
tp = 0            # True Positive
pred_total = 0    # 모델이 예측한 총 번호판 박스 수
pred_total_cars = 0 # 모델이 예측한 총 치량 수

for k, img_path in enumerate(image_files):
    if (k + 1) % 100 == 0:
        print(f"Processing [{k+1}/{total_images}] ...")

    # [최적화] 라벨링 파일(JSON)이 없으면 추론하지 않고 바로 패스 (시간 단축)
    json_file = img_path.replace('01.원천데이터', '02.라벨링데이터').replace('.jpg', '.json') 
    if not os.path.exists(json_file):
        continue

    img = cv2.imread(img_path)
    if img is None: continue
    
    h, w = img.shape[:2]
    h2, w2 = h // 2, w // 2
    m_h, m_w = int(h * MARGIN), int(w * MARGIN)

    # 1. 4분할 크롭 (차량 탐지용)
    crops_info = [
        (img[0:h2+m_h, 0:w2+m_w],      0,      0),      
        (img[0:h2+m_h, w2-m_w:w],      w2-m_w, 0),       
        (img[h2-m_h:h, 0:w2+m_w],      0,      h2-m_h), 
        (img[h2-m_h:h, w2-m_w:w],      w2-m_w, h2-m_h)  
    ]

    # 배치 추론 (차량 모델)
    batch_imgs = [c[0] for c in crops_info]
    results_car = car_model(batch_imgs, conf=CONF_CAR, verbose=False, device=DEVICE)

    # 차량 좌표 통합
    car_boxes_list = []
    car_scores_list = []
    
    for i, res in enumerate(results_car):
        if len(res.boxes) == 0: continue
        
        boxes = res.boxes.xyxy.cpu()
        boxes[:, [0, 2]] += crops_info[i][1] 
        boxes[:, [1, 3]] += crops_info[i][2] 
        
        car_boxes_list.append(boxes)
        car_scores_list.append(res.boxes.conf.cpu())

    final_cars = []
    if len(car_boxes_list) > 0:
        all_boxes = torch.cat(car_boxes_list)
        all_scores = torch.cat(car_scores_list)
        
        # NMS 적용 (차량)
        keep_indices = torchvision.ops.nms(all_boxes, all_scores, IOU_CAR)
        final_cars = all_boxes[keep_indices]
        
    pred_total_cars += len(final_cars)
    
    # 2. 크롭된 차량에서 번호판 탐지
    final_pred_plates = [] # 이 이미지의 최종 예측 번호판 박스들

    for car_box in final_cars:
        cx1, cy1, cx2, cy2 = map(int, car_box)

        # Padding 계산
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

        # 번호판 추론
        results_plate = plate_model(car_img_crop, conf=CONF_PLATE, verbose=False, device=DEVICE)

        for res in results_plate:
            for box in res.boxes:
                # 번호판 좌표 (Crop 기준)
                px1, py1, px2, py2 = map(int, box.xyxy[0])

                # 원본 이미지 기준으로 좌표 복원
                real_px1 = px1 + crop_x1
                real_py1 = py1 + crop_y1
                real_px2 = px2 + crop_x1
                real_py2 = py2 + crop_y1

                # 최종 예측 리스트에 추가 (x1, y1, x2, y2)
                final_pred_plates.append([real_px1, real_py1, real_px2, real_py2])

    # 모델이 예측한 번호판 박스 수 누적
    pred_total += len(final_pred_plates)

    # 3. Ground Truth 읽어오기 및 평가 로직
    gt_boxes = []
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        info = data.get("Learning_Data_Info", {})
        annotations = info.get("annotations", [])
        
        for ann in annotations:
            for lp in ann.get("license_plate", []):
                bbox = lp.get("bbox", [])
                if len(bbox) == 4:
                    x, y, bw, bh = bbox
                    # xywh -> xyxy 변환
                    gt_boxes.append([x, y, x + bw, y + bh])

    # GT vs Pred (IoU 계산)
    for num_gt in gt_boxes:
        gt_total += 1 
        is_detected = False
        
        for pred in final_pred_plates:
            # IoU가 0.5 이상이면 정답 처리
            if calculate_iou(num_gt, pred) >= IOU_EVAL:
                is_detected = True
                break 
        
        if is_detected:
            tp += 1

# ================= 결과 출력 =================
recall = (tp / gt_total) * 100 if gt_total > 0 else 0
precision = (tp / pred_total) * 100 if pred_total > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n[ 모델 A: 4분할 추론 성능 평가 ]")
print("===========================================")
print(f"모델 예측 차량 수(Pred) : {pred_total_cars} 개")
print("===========================================")
print(f"전체 정답 수 (GT)      : {gt_total} 개")
print(f"모델 예측 박스 수 (Pred): {pred_total} 개")
print(f"정답을 맞춘 개수 (TP)   : {tp} 개")
print("===========================================")
print(f"Precision (정밀도)     : {precision:.2f}%")
print(f"Recall (재현율)        : {recall:.2f}%")
print(f"F1 Score               : {f1_score:.2f}%")