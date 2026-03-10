import cv2
import os
import torch
import torchvision
from ultralytics import YOLO

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

# 출력 폴더 생성
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# 모델 로드
model = YOLO(MODEL_DIR)

# 비디오 캡처 객체 초기화
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: 동영상 파일을 열 수 없습니다.")
    exit()

# 비디오 속성 가져오기
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 비디오 저장 객체 초기화
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

print(f"총 {total_frames} 프레임 처리 시작 (4분할 추론)")

frame_count = 0

# 프레임 단위 반복문
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % 60 == 0:
        print(f"Processing... [{frame_count}/{total_frames}]")
        
    h2, w2 = h // 2, w // 2
    m_h, m_w = int(h * MARGIN), int(w * MARGIN)

    # Overlap 적용 (4분할)
    crops_info = [
        (frame[0:h2+m_h, 0:w2+m_w],      0,      0),      
        (frame[0:h2+m_h, w2-m_w:w],      w2-m_w, 0),       
        (frame[h2-m_h:h, 0:w2+m_w],      0,      h2-m_h), 
        (frame[h2-m_h:h, w2-m_w:w],      w2-m_w, h2-m_h)  
    ]

    # 배치 추론 (4장 한번에)
    batch_imgs = [c[0] for c in crops_info]
    results = model(batch_imgs, conf=CONF, verbose=False, device='0')

    all_boxes_list = []
    all_scores_list = []
    all_classes_list = [] 
    
    for i, res in enumerate(results):
        if len(res.boxes) == 0: continue
        
        # 좌표 보정
        boxes = res.boxes.xyxy.cpu()
        boxes[:, [0, 2]] += crops_info[i][1] 
        boxes[:, [1, 3]] += crops_info[i][2] 
        
        all_boxes_list.append(boxes)
        all_scores_list.append(res.boxes.conf.cpu())
        all_classes_list.append(res.boxes.cls.cpu()) # 클래스 저장

    if all_boxes_list:
        all_boxes = torch.cat(all_boxes_list)
        all_scores = torch.cat(all_scores_list)
        all_classes = torch.cat(all_classes_list) # 클래스 합치기

        # NMS 중복 제거
        keep_indices = torchvision.ops.nms(all_boxes, all_scores, IOU)

        for idx in keep_indices:
            x1, y1, x2, y2 = map(int, all_boxes[idx])
            score = float(all_scores[idx])
            cls_id = int(all_classes[idx])
            
            # 라벨 텍스트
            label_text = f"{model.names[cls_id]} {score:.2f}"

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

    # 추론과 그리기가 완료된 프레임을 동영상으로 저장
    out.write(frame)

# 자원 해제
cap.release()
out.release()
print("Inference Completed.")