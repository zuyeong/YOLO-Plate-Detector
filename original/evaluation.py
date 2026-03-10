import os
import glob
import json
from ultralytics import YOLO

def iou(box1, box2):
    # 교집합 넓이 
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 합집합 넓이
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # IoU 
    return intersection / union

model = YOLO('YOLOv11n.pt')
test_image_dir = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/Test/01.원천데이터"
image_file = glob.glob(os.path.join(test_image_dir, '**', '*.jpg'), recursive=True)

total_images = len(image_file)

print(f"검색된 총 이미지 개수: {total_images}장")

gt = 0         # 전체 정답 수
tp = 0               # 탐지 성공 수
iou_threshold = 0.5  # 성공 기준 (50%)
total_pred = 0

for images in image_file:
    results = model(images, conf=0.45, verbose=False)
    pred_boxes = results[0].boxes.xyxy.tolist()     # 첫 번째 사진 결과 -> 박스 좌표만 뽑기 -> 리스트 형태로 변환
    
    total_pred += len(pred_boxes)
    
    # 이름 같은 json 파일 경로 찾기
    json_file = images.replace('01.원천데이터', '02.라벨링데이터').replace('.jpg', '.json') 
    if not os.path.exists(json_file):
        continue
    
    # 정답 파일
    gt_boxes = []
    
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        info = data.get("Learning_Data_Info", {})
        annotations = info.get("annotations", [])
        
        for ann in annotations:
            for lp in ann.get("license_plate", []):
                # gt bbox 좌표 뽑기 
                bbox = lp.get("bbox", [])
                x, y, w, h = bbox
                
                # xywh -> xyxy 변환
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                gt_boxes.append([x1, y1, x2, y2])

    # 예측 box GT와 겹치는지 확인
    for num_gt in gt_boxes:
        gt += 1 
        is_detected = False
        
        for pred in pred_boxes:
            if iou(num_gt, pred) >= iou_threshold:  # 0.5 이상이면 성공
                is_detected = True
                break # 한 번 성공하면 끝 
        
        if is_detected:
            tp += 1
            
    
            


recall = (tp / gt) * 100
precision = (tp / total_pred) * 100

print(f"전체 정답 수(GT)          : {gt} 개")
print(f"모델이 예측한 박스 수 (Pred): {total_pred} 개") # 표의 'bbox 수'
print(f"정답을 맞춘 개수 (TP)      : {tp} 개")
print(f"========")
print(f"Precision : {precision:.2f}%") # 보고서 표 빈칸에 이 숫자를 적으세요
print(f"Recall    : {recall:.2f}%")
        
    