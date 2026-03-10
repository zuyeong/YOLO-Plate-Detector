import cv2
import os
import torch
import torchvision
import numpy as np
from ultralytics import YOLO

# ================= 설정 =================
VIDEO_PATH = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/original_video.mp4" 
OUTPUT_PATH = "/home/cjy/workspace/video/inference/hausdorff2.mp4" 
MODEL_DIR = "/home/cjy/workspace/original/YOLOv11n.pt"

MARGIN = 0.25   # 15% Overlap -> 25% (코드 기준)
CONF = 0.25     # 신뢰도
IOU = 0.3       # NMS 중복 제거 기준

BOX_COLOR = (255, 0, 0)    
TEXT_COLOR = (255, 255, 255)     
FONT_SCALE = 1.0          
THICKNESS = 2          
# ========================================

class HausdorffTracker:
    def __init__(self, delta=2.0, max_missed=30):
        self.tracks = []
        self.next_id = 1
        self.delta = delta        # 형태 변화 허용 임계값 (논문 최적값 2 적용)
        self.max_missed = max_missed # ID 유지 프레임 수

    def get_model_points(self, edges, bbox):
        """바운딩 박스 내의 윤곽선(Edge) 픽셀을 모델로 추출합니다."""
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(edges.shape[1], x2), min(edges.shape[0], y2)
        
        roi = edges[y1:y2, x1:x2]
        pts = np.column_stack(np.where(roi > 0)) # (y, x) 형태
        
        # 중심점 기준 상대 좌표로 변환
        if len(pts) > 0:
            cy, cx = (y2 - y1) // 2, (x2 - x1) // 2
            pts[:, 0] -= cy
            pts[:, 1] -= cx
            
        return pts, (x2 - x1, y2 - y1)

    def partial_hausdorff(self, dist_map, model_pts, cx, cy):
        """거리 맵을 이용하여 부분 하우스돌프 거리를 빠르게 계산합니다."""
        if len(model_pts) == 0:
            return float('inf')
        
        y_coords = model_pts[:, 0] + int(cy)
        x_coords = model_pts[:, 1] + int(cx)
        
        # 맵 범위를 벗어나는 점 필터링
        valid = (y_coords >= 0) & (y_coords < dist_map.shape[0]) & \
                (x_coords >= 0) & (x_coords < dist_map.shape[1])
        
        if not np.any(valid): return float('inf')
        
        distances = dist_map[y_coords[valid], x_coords[valid]]
        # 노이즈에 민감한 Max 대신 90백분위수를 사용하여 부분 하우스돌프 거리 구현
        return np.percentile(distances, 90)

    def search_2d_logarithmic(self, dist_map, model_pts, init_cx, init_cy, w, h):
        """2D-Logarithmic 기법으로 최소 하우스돌프 거리 위치를 탐색합니다."""
        step = max(w, h) // 4
        step = max(2, step)
        
        best_cx, best_cy = init_cx, init_cy
        best_dist = self.partial_hausdorff(dist_map, model_pts, best_cx, best_cy)
        
        while step >= 1:
            found_better = False
            # 9개 위치 (중앙 포함) 탐색
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0: continue
                    
                    cx, cy = best_cx + dx, best_cy + dy
                    dist = self.partial_hausdorff(dist_map, model_pts, cx, cy)
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_cx, best_cy = cx, cy
                        found_better = True
            
            # 더 나은 위치를 찾지 못하면 탐색 반경(step) 축소
            if not found_better:
                step //= 2
                
        return best_cx, best_cy, best_dist

    def update_tracks(self, frame, detections):
        """프레임과 YOLO 탐지 결과를 받아 트랙을 업데이트합니다."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 연산 속도 향상을 위한 거리 맵(Distance Map) 생성
        inv_edges = cv2.bitwise_not(edges)
        dist_map = cv2.distanceTransform(inv_edges, cv2.DIST_L2, 3)
        
        matched_det_indices = set()
        active_tracks = []
        
        # 1. 기존 트랙 갱신 (2D-Log 탐색 및 YOLO 박스 보정)
        for track in self.tracks:
            if track['missed'] > self.max_missed: continue
            
            x1, y1, x2, y2 = track['bbox']
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            w, h = x2 - x1, y2 - y1
            
            # 논문 기반 탐색 로직
            new_cx, new_cy, dist = self.search_2d_logarithmic(dist_map, track['model_pts'], cx, cy, w, h)
            
            if dist <= self.delta: # 형태가 유사한 경우 (정합 성공)
                nx1, ny1 = int(new_cx - w/2), int(new_cy - h/2)
                nx2, ny2 = int(new_cx + w/2), int(new_cy + h/2)
                track['bbox'] = [nx1, ny1, nx2, ny2]
                track['missed'] = 0
                
                # YOLO Detections와 IoU 기반 보정 (의미론적 클래스 정보 유지)
                best_iou, best_j = 0, -1
                for j, (det_box, _, _) in enumerate(detections):
                    if j in matched_det_indices: continue
                    dx1, dy1, dw, dh = det_box
                    dx2, dy2 = dx1 + dw, dy1 + dh
                    
                    ix1, iy1 = max(nx1, dx1), max(ny1, dy1)
                    ix2, iy2 = min(nx2, dx2), min(ny2, dy2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    union = (w * h) + (dw * dh) - inter
                    iou = inter / union if union > 0 else 0
                    
                    if iou > 0.3 and iou > best_iou:
                        best_iou = iou
                        best_j = j
                        
                if best_j != -1:
                    matched_det_indices.add(best_j)
                    dx1, dy1, dw, dh = detections[best_j][0]
                    track['bbox'] = [dx1, dy1, dx1 + dw, dy1 + dh]
                
                # 모델 갱신 (추적 마스크 재구성)
                pts, _ = self.get_model_points(edges, track['bbox'])
                if len(pts) > 0:
                    track['model_pts'] = pts
                    
            else:
                track['missed'] += 1

        # 2. 매칭되지 않은 새로운 YOLO 객체 등록
        for j, (det_box, score, cls_id) in enumerate(detections):
            if j in matched_det_indices: continue
            dx1, dy1, dw, dh = map(int, det_box)
            bbox = [dx1, dy1, dx1 + dw, dy1 + dh]
            
            pts, _ = self.get_model_points(edges, bbox)
            if len(pts) > 0:
                self.tracks.append({
                    'id': self.next_id,
                    'bbox': bbox,
                    'model_pts': pts,
                    'cls_id': cls_id,
                    'missed': 0
                })
                self.next_id += 1
        
        # 삭제 프레임 초과된 오래된 트랙 정리
        self.tracks = [t for t in self.tracks if t['missed'] <= self.max_missed]
        
        return [t for t in self.tracks if t['missed'] == 0]

# 메인 실행부
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

# 커스텀 모델 기반 트래커 초기화
tracker = HausdorffTracker(delta=2.0, max_missed=30)

print(f"총 {total_frames} 프레임 처리 시작 (4분할 추론 + Hausdorff Tracker)")

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

    # 하우스돌프 트래커 업데이트
    active_tracks = tracker.update_tracks(frame, detections)

    for track in active_tracks:
        display_id = track['id']
        x1, y1, x2, y2 = track['bbox']
        cls_id = track['cls_id']
        
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
        cv2.putText(frame, label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, THICKNESS)

    out.write(frame)

cap.release()
out.release()
print("Inference Completed.")