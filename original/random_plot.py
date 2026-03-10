import os
import glob
import random
from pathlib import Path
import cv2 # pyright: ignore[reportMissingImports]

train_image_dir = "/home/cjy/workspace/original/processed_original/images/train"
train_label_dir = "/home/cjy/workspace/original/processed_original/labels/train"
plot_dir = "/home/cjy/workspace/original/processed_original/plotted_image"

SAMPLES = 10

def plot(n):
    imgs = glob.glob(os.path.join(train_image_dir, "*.png"))
    samples = random.sample(imgs, n) # 랜덤 선택
    
    for img_path in samples:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        stem = Path(img_path).stem # 파일명 뽑기기
        label_path = os.path.join(train_label_dir, stem + ".txt")
        
        # [0 x_center y_center w_norm h_norm] 쪼개기
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                cls = parts[0]
                xc, yc, wn, hn = map(float, parts[1:])
                
                # 좌상단(x1,y1), 우하단(x2,y2)
                x1 = int((xc - wn/2) * w)
                y1 = int((yc - hn/2) * h)
                x2 = int((xc + wn/2) * w)
                y2 = int((yc + hn/2) * h)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1) # bbox 그리기

        plot_path = os.path.join(plot_dir, stem + ".png")
        cv2.imwrite(plot_path, img)  
                
random.seed()
plot(SAMPLES)               