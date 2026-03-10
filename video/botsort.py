import cv2
import os
import torch
import torchvision
from ultralytics import YOLO
import supervision as sv  

# ================= 설정 =================
VIDEO_PATH = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/original_video.mp4" 
OUTPUT_PATH = "/home/cjy/workspace/video/inference/botsort_quarter.mp4" 
MODEL_DIR = "/home/cjy/workspace/original/YOLOv11n.pt"

MARGIN = 0.15   # 15% Overlap
CONF = 0.45     # 신뢰도
IOU = 0.3       # NMS 중복 제거 기준

BOX_COLOR = (255, 0, 0)    
TEXT_COLOR = (255, 255, 255)     
FONT_SCALE = 1.0          
THICKNESS = 2          
# ========================================

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
model = YOLO(MODEL_DIR)

tracker = sv.ByteTrack()

cap = cv2.VideoCapture(VIDEO_PATH)
1221
if not cap.isOpened():
    print("Error: 동영상 파일을 열 수 없습니다.")
    exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

print(f"총 {total_frames} 프레임 처리 시작 (4분할 + ID 추적)")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % 60 == 0:
        print(f"Processing... [{frame_count}/{total_frames}]")
        
    h2, w2 = h // 2, w // 2
    m_h, m_w = int(h * MARGIN), int(w * MARGIN)

    crops_info = [
        (frame[0:h2+m_h, 0:w2+m_w],      0,      0),      
        (frame[0:h2+m_h, w2-m_w:w],      w2-m_w, 0),       
        (frame[h2-m_h:h, 0:w2+m_w],      0,      h2-m_h), 
        (frame[h2-m_h:h, w2-m_w:w],      w2-m_w, h2-m_h)  
    ]

    batch_imgs = [c[0] for c in crops_info]
    results = model(batch_imgs, conf=CONF, verbose=False, device='0')

    all_boxes_list = []
    all_scores_list = []
    all_classes_list = [] 
    
    for i, res in enumerate(results):
        if len(res.boxes) == 0: continue
        
        boxes = res.boxes.xyxy.cpu()
        boxes[:, [0, 2]] += crops_info[i][1] 
        boxes[:, [1, 3]] += crops_info[i][2] 
        
        all_boxes_list.append(boxes)
        all_scores_list.append(res.boxes.conf.cpu())
        all_classes_list.append(res.boxes.cls.cpu()) 

    if all_boxes_list:
        all_boxes = torch.cat(all_boxes_list)
        all_scores = torch.cat(all_scores_list)
        all_classes = torch.cat(all_classes_list) 

        # NMS 중복 제거
        keep_indices = torchvision.ops.nms(all_boxes, all_scores, IOU)

        # 👈 추가 3: NMS를 통과한 결과만 모아서 supervision 형식으로 변환
        final_boxes = all_boxes[keep_indices].numpy()
        final_scores = all_scores[keep_indices].numpy()
        final_classes = all_classes[keep_indices].numpy().astype(int)

        detections = sv.Detections(
            xyxy=final_boxes,
            confidence=final_scores,
            class_id=final_classes
        )

        # 👈 추가 4: 추적기에 데이터를 넣고 ID가 포함된 결과 받아오기
        tracked_detections = tracker.update_with_detections(detections)

        # 추적된 결과(tracked_detections)를 순회하며 박스와 ID 그리기
        for i in range(len(tracked_detections.xyxy)):
            x1, y1, x2, y2 = map(int, tracked_detections.xyxy[i])
            score = tracked_detections.confidence[i]
            cls_id = tracked_detections.class_id[i]
            tracker_id = tracked_detections.tracker_id[i] # 드디어 고유 ID 획득!
            
            # 라벨 텍스트에 ID 추가
            label_text = f"ID:{tracker_id} {model.names[cls_id]} {score:.2f}"

            # 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 3)

            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS)
            if y1 - th - 10 < 0:
                cv2.rectangle(frame, (x1, y1), (x1 + tw, y1 + th + 10), BOX_COLOR, -1)
                text_y = y1 + th + 5
            else:
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), BOX_COLOR, -1)
                text_y = y1 - 5
            cv2.putText(frame, label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, THICKNESS)

    out.write(frame)

cap.release()
out.release()
print("Inference Completed.")