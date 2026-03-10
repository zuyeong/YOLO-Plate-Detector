import cv2
import os
import math  
import torch
import torchvision
from ultralytics import YOLO

# ================= 설정 =================
VIDEO_PATH = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/original_video.mp4"
OUTPUT_PATH = "/home/cjy/workspace/video/inference/center_dist2.mp4"
MODEL_DIR = "/home/cjy/workspace/original/YOLOv11n.pt"

MARGIN = 0.15
CONF = 0.6
IOU = 0.3

# 트래킹(=ID 유지) 관련
MATCH_IOU = 0.12        # 낮춰서 끊김 방지(0.10~0.20 추천)
MAX_DIST = 200          # 추가됨: 중심점 거리 허용 임계값(픽셀 단위). 이 거리 안이면 같은 객체로 추적 시도
MAX_AGE = 20            # 잠깐 가려져도 유지(늘릴수록 유지 잘됨)
SMOOTH = 0.60           # bbox 스무딩 (0.7~0.9 추천)

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)
FONT_SCALE = 1.0
THICKNESS = 2
# ========================================

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

# 두 bounding box의 중심점 거리를 계산하는 함수
def center_distance(a, b):
    ax_c = (a[0] + a[2]) / 2.0
    ay_c = (a[1] + a[3]) / 2.0
    bx_c = (b[0] + b[2]) / 2.0
    by_c = (b[1] + b[3]) / 2.0
    return math.hypot(ax_c - bx_c, ay_c - by_c)

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

print(f"총 {total_frames} 프레임 처리 시작 (4분할 추론 + ID Tracking)")

# ================== 간단 트래커 상태 ==================
# tid -> {"bbox": [x1,y1,x2,y2] float, "cls": int, "age": int}
tracks = {}
next_tid = 1
# =======================================================

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
        if len(res.boxes) == 0:
            continue

        boxes = res.boxes.xyxy.cpu()
        boxes[:, [0, 2]] += crops_info[i][1]
        boxes[:, [1, 3]] += crops_info[i][2]

        all_boxes_list.append(boxes)
        all_scores_list.append(res.boxes.conf.cpu())
        all_classes_list.append(res.boxes.cls.cpu())

    dets = []
    if all_boxes_list:
        all_boxes = torch.cat(all_boxes_list)
        all_scores = torch.cat(all_scores_list)
        all_classes = torch.cat(all_classes_list)

        keep_indices = torchvision.ops.nms(all_boxes, all_scores, IOU)

        for idx in keep_indices:
            x1, y1, x2, y2 = map(float, all_boxes[idx])  # float 유지(스무딩 때문에)
            cls_id = int(all_classes[idx])
            score = float(all_scores[idx])
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append({"bbox": [x1, y1, x2, y2], "cls": cls_id, "score": score})
            
    # ================== 한 차량 내 오탐지(로고 등) 중복 제거 ==================
    filtered_dets = []
    for d in dets:
        is_duplicate = False
        for fd in filtered_dets:
            # 두 박스의 중심점 거리가 250 픽셀 이내면 같은 차의 오탐지로 간주 (필요시 수치 조절)
            if center_distance(d["bbox"], fd["bbox"]) < 250:
                is_duplicate = True
                # 둘 중 모델이 더 확신하는(score가 높은) 박스로 덮어씌움
                if d["score"] > fd["score"]:
                    fd["bbox"] = d["bbox"]
                    fd["cls"] = d["cls"]
                    fd["score"] = d["score"]
                break
        if not is_duplicate:
            filtered_dets.append(d)
    
    dets = filtered_dets  # 중복이 제거된 깔끔한 결과만 트래커로 넘김
    # ====================================================================================

    # ================== 트래킹(DeepSORT 없이) ==================
    # 1) 모든 트랙 age 증가 및 만료 제거
    for tid in list(tracks.keys()):
        tracks[tid]["age"] += 1
        if tracks[tid]["age"] > MAX_AGE:
            del tracks[tid]

    det_assigned_tid = [-1] * len(dets)
    used_tids = set()

    # 2) IoU + 중심점 거리 매칭 (⭐ 수정된 부분)
    if dets and tracks:
        candidates = []
        for di, d in enumerate(dets):
            for tid, t in tracks.items():
                iou = iou_xyxy(t["bbox"], d["bbox"])
                dist = center_distance(t["bbox"], d["bbox"])

                # IoU가 기준 이상이거나 거리가 MAX_DIST 이하인 경우 후보 등록
                if iou >= MATCH_IOU or dist <= MAX_DIST:
                    # 결합 점수 생성: IoU는 높을수록(최대 1), 거리는 짧을수록(최대 1) 높은 점수 부여
                    dist_score = max(0.0, 1.0 - (dist / MAX_DIST))
                    combined_score = iou + dist_score
                    candidates.append((combined_score, tid, di))

        # 결합 점수를 기준으로 내림차순 정렬 (점수가 가장 높은 쌍부터 매칭)
        candidates.sort(reverse=True, key=lambda x: x[0])

        for score, tid, di in candidates:
            if det_assigned_tid[di] != -1:
                continue
            if tid in used_tids:
                continue

            det_assigned_tid[di] = tid
            used_tids.add(tid)

            # ⭐ bbox 스무딩으로 흔들림 감소 → 다음 프레임 매칭 더 잘 됨
            pb = tracks[tid]["bbox"]
            db = dets[di]["bbox"]
            tracks[tid]["bbox"] = [
                SMOOTH * db[0] + (1 - SMOOTH) * pb[0],
                SMOOTH * db[1] + (1 - SMOOTH) * pb[1],
                SMOOTH * db[2] + (1 - SMOOTH) * pb[2],
                SMOOTH * db[3] + (1 - SMOOTH) * pb[3],
            ]
            # 클래스는 “현재 탐지값”으로 업데이트
            tracks[tid]["cls"] = dets[di]["cls"]
            tracks[tid]["age"] = 0

    # 3) 매칭 실패 det는 새 트랙 생성 (ID 1부터 순차)
    for di, d in enumerate(dets):
        if det_assigned_tid[di] == -1:
            tid = next_tid
            next_tid += 1
            tracks[tid] = {"bbox": d["bbox"], "cls": d["cls"], "age": 0}
            det_assigned_tid[di] = tid

    # ================== 그리기 (score 제거, ID 표시) ==================
    for di, d in enumerate(dets):
        tid = det_assigned_tid[di]
        cls_id = tracks[tid]["cls"]
        x1, y1, x2, y2 = tracks[tid]["bbox"]  # 스무딩된 bbox로 그림
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        label_text = f"ID {tid} {model.names[cls_id]}"

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