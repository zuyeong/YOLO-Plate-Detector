import cv2
import glob
import os
import torch
import torchvision
from ultralytics import YOLO


IMG_DIR = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/Test/01.원천데이터/**/*.jpg"
OUTPUT_DIR = "/home/cjy/workspace/original_quarter/inference_image"
MODEL_DIR = "/home/cjy/workspace/original/YOLOv11n.pt"

MARGIN = 0.15   # 15% Overlap
CONF = 0.45     # 신뢰도
IOU = 0.3     # NMS 중복 제거 기준

BOX_COLOR = (255, 0, 0)    
TEXT_COLOR = (255, 255, 255)     
FONT_SCALE = 1.0          
THICKNESS = 2           

os.makedirs(OUTPUT_DIR, exist_ok=True)
model = YOLO(MODEL_DIR)

img_paths = glob.glob(IMG_DIR, recursive=True)
total_images = len(img_paths)

print(f"검색된 총 이미지 개수: {total_images}장")
print(f"Inference Start")


for k, img_path in enumerate(img_paths):
    if (k + 1) % 100 == 0:
        print(f"Processing [{k+1}/{total_images}]")
        
    img = cv2.imread(img_path)
    if img is None: continue
    
    h, w = img.shape[:2]
    h2, w2 = h // 2, w // 2
    m_h, m_w = int(h * MARGIN), int(w * MARGIN)

    # Overlap 적용 (4분할)
    crops_info = [
        (img[0:h2+m_h, 0:w2+m_w],      0,      0),      
        (img[0:h2+m_h, w2-m_w:w],      w2-m_w, 0),       
        (img[h2-m_h:h, 0:w2+m_w],      0,      h2-m_h), 
        (img[h2-m_h:h, w2-m_w:w],      w2-m_w, h2-m_h)  
    ]

    # 배치 추론 (4장 한번에)
    batch_imgs = [c[0] for c in crops_info]
    results = model(batch_imgs, conf=CONF, verbose=False, device='3')

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
            
            # 라벨 텍스트(license_plate 0.95)
            label_text = f"{model.names[cls_id]} {score:.2f}"

            # 박스 그리기
            cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 3)

            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS)
            if y1 - th - 10 < 0:
                cv2.rectangle(img, (x1, y1), (x1 + tw, y1 + th + 10), BOX_COLOR, -1)
                text_y = y1 + th + 5
            else:
                cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1), BOX_COLOR, -1)
                text_y = y1 - 5
            cv2.putText(img, label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, THICKNESS)


    filename = os.path.basename(img_path)
    save_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(save_path, img)

print("Inference Completed.")