import cv2  # pyright: ignore[reportMissingImports]
import os
import glob
import torch  # pyright: ignore[reportMissingImports]
from pathlib import Path

device_num = 0
if torch.cuda.is_available():
    torch.cuda.set_device(device_num) 
    device = f"cuda:{device_num}"
else:
    device = "cpu"
print(device)

raw_dir = [
    "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/data/Train/01.원천데이터/**/*.jpg",
    "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/data/Validation/01.원천데이터/**/*.jpg"
]

train_dir = "/home/cjy/workspace/original/processed_original/images/train"
val_dir = "/home/cjy/workspace/original/processed_original/images/validation"

count = 0

for path in raw_dir:
    if "Train" in path:
        resize_dir = train_dir
    else:
        resize_dir = val_dir
        
    for file in glob.iglob(path, recursive=True):
        img = cv2.imread(file)
        if img is None:
            continue

        h, w = img.shape[:2]

        # 긴 쪽 기준으로 640 맞추기
        scale = 640 / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        png_name = Path(file).stem + ".png"
        resize_path = os.path.join(resize_dir, png_name) # 새로운 경로에 파일 이름 합치기
        cv2.imwrite(resize_path, resized_img)

        count += 1
        if count % 1000 == 0:
            print(f"현재 {count}장 처리 완료")

print(f"최종 완료: 총 {count}장 처리됨")
