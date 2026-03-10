import cv2
import glob
import os
import torch
import torchvision
import json
from ultralytics import YOLO

MODEL_DIR = "/home/cjy/workspace/original/YOLOv11n.pt"
TEST_IMAGE_DIR = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/Test/01.원천데이터"

MARGIN = 0.15   # 15% Overlap
CONF = 0.45     # 신뢰도 
IOU_NMS = 0.4  # 4분할 합칠 때 중복 제거 기준
IOU_EVAL = 0.5  # 정답으로 인정할 IoU 기준 (0.5 이상이면 정답)

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

model = YOLO(MODEL_DIR)
image_files = glob.glob(os.path.join(TEST_IMAGE_DIR, '**', '*.jpg'), recursive=True)
total_images = len(image_files)

print(f"검색된 총 이미지 개수: {total_images}장")
print(f"4-Split Evaluation Start...")

gt_total = 0      # Ground Truth
tp = 0            # True Positive
pred_total = 0    # 모델이 예측한 총 박스 수

for k, img_path in enumerate(image_files):

    if (k + 1) % 100 == 0:
        print(f"Processing [{k+1}/{total_images}]")

    img = cv2.imread(img_path)
    if img is None: continue
    
    h, w = img.shape[:2]
    h2, w2 = h // 2, w // 2
    m_h, m_w = int(h * MARGIN), int(w * MARGIN)

    # 4분할 크롭 정보
    crops_info = [
        (img[0:h2+m_h, 0:w2+m_w],      0,      0),      
        (img[0:h2+m_h, w2-m_w:w],      w2-m_w, 0),       
        (img[h2-m_h:h, 0:w2+m_w],      0,      h2-m_h), 
        (img[h2-m_h:h, w2-m_w:w],      w2-m_w, h2-m_h)  
    ]

    # 배치 추론
    batch_imgs = [c[0] for c in crops_info]
    results = model(batch_imgs, conf=CONF, verbose=False, device='3', augment=True)

    # 좌표 복원 및 합치기
    all_boxes_list = []
    all_scores_list = []
    
    for i, res in enumerate(results):
        if len(res.boxes) == 0: continue
        
        # 좌표 보정 (Offset 적용)
        boxes = res.boxes.xyxy.cpu()
        boxes[:, [0, 2]] += crops_info[i][1] 
        boxes[:, [1, 3]] += crops_info[i][2] 
        
        all_boxes_list.append(boxes)
        all_scores_list.append(res.boxes.conf.cpu())

    final_pred_boxes = [] # 이 이미지의 최종 예측 박스들

    # 검출된 박스가 있을 경우 NMS 적용
    if all_boxes_list:
        all_boxes = torch.cat(all_boxes_list)
        all_scores = torch.cat(all_scores_list)

        # NMS 수행 (중복 박스 제거)
        keep_indices = torchvision.ops.nms(all_boxes, all_scores, IOU_NMS)
        
        # 최종 남은 박스들만 리스트로 변환
        final_pred_boxes = all_boxes[keep_indices].tolist()

    # 모델이 예측한 박스 수 누적
    pred_total += len(final_pred_boxes)

    json_file = img_path.replace('01.원천데이터', '02.라벨링데이터').replace('.jpg', '.json') 
    if not os.path.exists(json_file):
        continue
    
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

    # GT vs Pred
    for num_gt in gt_boxes:
        gt_total += 1 
        is_detected = False
        
        for pred in final_pred_boxes:
            # IoU가 0.5 이상이면 정답 처리
            if calculate_iou(num_gt, pred) >= IOU_EVAL:
                is_detected = True
                break 
        
        if is_detected:
            tp += 1


recall = (tp / gt_total) * 100 if gt_total > 0 else 0
precision = (tp / pred_total) * 100 if pred_total > 0 else 0

print(f"[ 4분할 추론 성능 평가 ]")
print(f"총 이미지 수           : {total_images} 장")
print("===========================================")
print(f"전체 정답 수 (GT)      : {gt_total} 개")
print(f"모델 예측 박스 수 (Pred): {pred_total} 개")
print(f"정답을 맞춘 개수 (TP)   : {tp} 개")
print("===========================================")
print(f"Precision (정밀도)     : {precision:.2f}%")
print(f"Recall (재현율)        : {recall:.2f}%")