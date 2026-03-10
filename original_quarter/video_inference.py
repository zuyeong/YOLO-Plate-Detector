import cv2
import torch
import torchvision
from ultralytics import YOLO

VIDEO_PATH = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/data/sample.mp4" 
OUTPUT_PATH = "/home/cjy/workspace/original_quarter/video.mp4" # 저장할 영상 경로
MODEL_DIR = "/home/cjy/workspace/original/YOLOv11n.pt"

MARGIN = 0.15   # 15% Overlap
CONF = 0.45     # 신뢰도
IOU = 0.3       # NMS 중복 제거 기준

BOX_COLOR = (255, 0, 0)     
TEXT_COLOR = (255, 255, 255)     
FONT_SCALE = 1.0          
THICKNESS = 2           
DEVICE = '0' 

model = YOLO(MODEL_DIR)

# 비디오 읽기 설정
cap = cv2.VideoCapture(VIDEO_PATH)
# 영상 정보 가져오기 
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 비디오 저장 설정 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
frame_idx = 0 # 현재 프레임 수

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    if frame_idx % 50 == 0:
        print(f"Processing [{frame_idx}/{total_frames}]")


    h, w = frame.shape[:2]
    h2, w2 = h // 2, w // 2
    m_h, m_w = int(h * MARGIN), int(w * MARGIN)

    crops_info = [
        (frame[0:h2+m_h, 0:w2+m_w],      0,      0),      # 좌상
        (frame[0:h2+m_h, w2-m_w:w],      w2-m_w, 0),      # 우상
        (frame[h2-m_h:h, 0:w2+m_w],      0,      h2-m_h), # 좌하
        (frame[h2-m_h:h, w2-m_w:w],      w2-m_w, h2-m_h)  # 우하
    ]

    # 배치 추론 (4장 한번에)
    batch_imgs = [c[0] for c in crops_info]
    results = model(batch_imgs, conf=CONF, verbose=False, device=DEVICE)

    all_boxes_list = []
    all_scores_list = []
    all_classes_list = [] 
    
    for i, res in enumerate(results):
        if len(res.boxes) == 0: continue
        
        # 좌표 보정 (Crop -> Original)
        boxes = res.boxes.xyxy.cpu()
        boxes[:, [0, 2]] += crops_info[i][1] # x offset
        boxes[:, [1, 3]] += crops_info[i][2] # y offset
        
        all_boxes_list.append(boxes)
        all_scores_list.append(res.boxes.conf.cpu())
        all_classes_list.append(res.boxes.cls.cpu())

    # NMS 
    if all_boxes_list:
        all_boxes = torch.cat(all_boxes_list)
        all_scores = torch.cat(all_scores_list)
        all_classes = torch.cat(all_classes_list)
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
            
            # 텍스트가 화면 위로 넘어가지 않게
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
print(f"Inference Completed. Saved to: {OUTPUT_PATH}")