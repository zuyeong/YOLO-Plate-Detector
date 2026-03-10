import os
import cv2
from pathlib import Path

train_image_dir = "/mnt/hdd_6tb/cjy/processed_quarter/images/train"
train_label_dir = "/mnt/hdd_6tb/cjy/processed_quarter/labels/train"
plot_dir = "/home/cjy/workspace/quarter/processed_quarter/plotted_image"


def plot_specific_files():
    base_name = "C-220920_10_SR13_01_N4615"
    for i in range(4):
        file_name = f"{base_name}_{i}.png"
        img_path = os.path.join(train_image_dir, file_name)
 
        img = cv2.imread(img_path)


        h, w = img.shape[:2]
        stem = Path(img_path).stem 
        label_path = os.path.join(train_label_dir, stem + ".txt")
        
        # 라벨 파일이 있는지 확인
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    xc, yc, wn, hn = map(float, parts[1:])
                    
                    # 좌상단(x1,y1), 우하단(x2,y2) 좌표 변환
                    x1 = int((xc - wn/2) * w)
                    y1 = int((yc - hn/2) * h)
                    x2 = int((xc + wn/2) * w)
                    y2 = int((yc + hn/2) * h)
                    
                    # bbox 그리기
                    cv2.rectangle(img, (x1, y1), (x2, y2), (225, 0, 0), 2)
        else:
            print(f"라벨 파일이 없음음: {label_path}")

        # 결과 저장
        plot_path = os.path.join(plot_dir, stem + ".png")
        cv2.imwrite(plot_path, img)

# 실행
if __name__ == "__main__":
    plot_specific_files()