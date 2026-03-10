import os
import random
import cv2

# 경로 설정
train_image_dir = "/mnt/hdd_6tb/cjy/processed_carcrop/images/train"
train_label_dir = "/mnt/hdd_6tb/cjy/processed_carcrop/labels/train"
plot_dir = "/home/cjy/workspace/car_detection/plotted_image"

os.makedirs(plot_dir, exist_ok=True)

# 파일이 많아도 listdir은 비교적 빠름
all_images = [f for f in os.listdir(train_image_dir) if f.endswith('.png')]

# 랜덤으로 10개
sampled_images = random.sample(all_images, min(10, len(all_images)))

for img_file in sampled_images:
    image_path = os.path.join(train_image_dir, img_file)
    label_file = img_file.replace('.png', '.txt')
    label_path = os.path.join(train_label_dir, label_file)

    # 이미지 읽기
    img = cv2.imread(image_path)
    if img is None:
        continue
    
    h, w, _ = img.shape

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            # YOLO format: class x_center y_center width height (normalized 0~1)
            # 필요하다면 class_id = int(parts[0]) 등을 사용
            
            # 좌표 변환 (Normalized -> Pixel)
            cx, cy, bw, bh = map(float, parts[1:5])
            
            # 좌상단, 우하단 좌표 계산
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            
            # 사각형 그리기 (초록색, 두께 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    save_path = os.path.join(plot_dir, img_file)
    cv2.imwrite(save_path, img)
    print(f"Saved: {save_path}")

print("완료!")