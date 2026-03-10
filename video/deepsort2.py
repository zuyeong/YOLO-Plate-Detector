import cv2
import os
import torch
import torchvision
import heapq 
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ================= 설정 =================
VIDEO_PATH = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/original_video.mp4"
OUTPUT_PATH = "/home/cjy/workspace/video/inference/deepsort2_1.mp4"
MODEL_DIR = "/home/cjy/workspace/original/YOLOv11n.pt"

MARGIN = 0.25
CONF = 0.25
IOU = 0.3

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

tracker = DeepSort(
    max_age=30,
    n_init=1,              
    max_iou_distance=0.99,
    max_cosine_distance=0.5
)

print(f"총 {total_frames} 프레임 처리 시작 (4분할 추론 + DeepSORT)")

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
    results = model(batch_imgs, conf=CONF, verbose=False, device=0)

    all_boxes_list = []
    all_scores_list = []
    all_classes_list = []

    for i, res in enumerate(results):
        if len(res.boxes) == 0:
            continue

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

    tracks = tracker.update_tracks(detections, frame=frame)

    # 이번 프레임에 살아있는 raw_id 집합
    alive_raw_ids = set(t.track_id for t in tracks)

    for track in tracks:
        # confirmed 트랙만 “숫자 ID”를 부여/표시 (tentative는 ID 소비 X)
        if not track.is_confirmed():
            continue

        raw_track_id = track.track_id

        # 커스텀 ID: 반납된 번호가 있으면 재사용, 없으면 새로 증가
        if raw_track_id not in custom_id_map:
            if free_custom_ids:
                custom_id_map[raw_track_id] = heapq.heappop(free_custom_ids)
            else:
                custom_id_map[raw_track_id] = next_custom_id
                next_custom_id += 1

        display_id = custom_id_map[raw_track_id]

        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        cls_id = track.get_det_class()
        class_name = model.names[cls_id] if cls_id is not None else "Unknown"
        label_text = f"ID:{display_id} {class_name}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 3)

        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS)
        if y1 - th - 10 < 0:
            cv2.rectangle(frame, (x1, y1), (x1 + tw, y1 + th + 10), BOX_COLOR, -1)
            text_y = y1 + th + 5
        else:
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), BOX_COLOR, -1)
            text_y = y1 - 5

        cv2.putText(frame, label_text, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, THICKNESS)

    out.write(frame)

cap.release()
out.release()
print("Inference Completed.")