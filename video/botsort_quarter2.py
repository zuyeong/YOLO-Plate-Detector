import cv2
import os
import torch
import torchvision
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ================= 설정 =================
VIDEO_PATH = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/original_video.mp4" 
OUTPUT_PATH = "/home/cjy/workspace/video/inference/deepsort12.mp4" 
MODEL_DIR = "/home/cjy/workspace/original/YOLOv11n.pt"

MARGIN = 0.25   # 15% Overlap
CONF = 0.25    # 신뢰도
IOU = 0.3       # NMS 중복 제거 기준

BOX_COLOR = (255, 0, 0)    
TEXT_COLOR = (255, 255, 255)     
FONT_SCALE = 1.0          
THICKNESS = 2          
# ========================================

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
model = YOLO(MODEL_DIR)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: 동영상 파일을 열 수 없습니다.")
    exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

# tracker = DeepSort(max_age=30, n_init=3)

# tracker = DeepSort(
#     max_age=30, 
#     n_init=2,                
#     max_iou_distance=0.9,   
#     max_cosine_distance=0.4  
# )

tracker = DeepSort(
    max_age=30,              # 번호판 가려졌을 때 30프레임(1초) 기억 유지 
    n_init=1,                # 1프레임만 탐지돼도 즉시 ID 부여
    max_iou_distance=0.99,   # 박스가 아주 조금만 겹쳐도, 심지어 안 겹쳐도 일단 같은 객체로 봐줌
    max_cosine_distance=0.5  # 외형 일치 
)

print(f"총 {total_frames} 프레임 처리 시작 (4분할 추론 + DeepSORT)")

# ID 점프 방지
custom_id_map = {}
next_custom_id = 1

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

    detections = []

    if all_boxes_list:
        all_boxes = torch.cat(all_boxes_list)
        all_scores = torch.cat(all_scores_list)
        all_classes = torch.cat(all_classes_list) 

        keep_indices = torchvision.ops.nms(all_boxes, all_scores, IOU)

        for idx in keep_indices:
            x1, y1, x2, y2 = map(int, all_boxes[idx])
            score = float(all_scores[idx])
            cls_id = int(all_classes[idx])
            
            w_box = x2 - x1
            h_box = y2 - y1
            detections.append(([x1, y1, w_box, h_box], score, cls_id))

    # 빈 리스트라도 매 프레임 트래커를 업데이트해야 기존 객체의 수명(max_age)이 정상 계산됩니다.
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        # ★ 수정 1: 확정(Confirmed)되지 않은 가짜/찰나의 트랙은 절대 번호를 주지 않고 무시!
        if not track.is_confirmed():
            continue
        
        raw_track_id = track.track_id # DeepSORT가 부여한 원본 ID
        
        # ★ 수정 2: 진짜 차로 확정되었을 때만 예쁜 순차 번호를 1개씩 발급합니다.
        if raw_track_id not in custom_id_map:
            custom_id_map[raw_track_id] = next_custom_id
            next_custom_id += 1
            
        display_id = custom_id_map[raw_track_id] # 화면에 표시할 1번부터 시작하는 깔끔한 ID

        ltrb = track.to_ltrb()    
        x1, y1, x2, y2 = map(int, ltrb)
        cls_id = track.get_det_class()
        
        # ... (이하 class_name, cv2.rectangle 등 화면에 그리는 코드는 기존과 동일하게 유지) ...
        

        
        class_name = model.names[cls_id] if cls_id is not None else "Unknown"
        
        # 텍스트에 커스텀 ID 추가
        label_text = f"ID:{display_id} {class_name}"

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