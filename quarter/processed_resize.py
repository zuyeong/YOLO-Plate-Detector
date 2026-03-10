import cv2  # pyright: ignore[reportMissingImports]
import os
import glob
from pathlib import Path

raw_dir = [
    "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/data/Train/01.원천데이터/**/*.jpg",
    "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/data/Validation/01.원천데이터/**/*.jpg"
]

train_dir = "/mnt/hdd_6tb/cjy/processed_quarter/images/train"
val_dir = "/mnt/hdd_6tb/cjy/processed_quarter/images/validation"

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
        mid_h, mid_w = h // 2, w // 2

        # 4분할 영역 
        crops = [
            img[0:mid_h, 0:mid_w],      # 좌측 상단 
            img[0:mid_h, mid_w:w],      # 우측 상단
            img[mid_h:h, 0:mid_w],      # 좌측 하단
            img[mid_h:h, mid_w:w]       # 우측 하단 
        ]

        base_name = Path(file).stem  # 원래 파일 이름 (확장자 제외)

        for i, crop in enumerate(crops):
            curr_h, curr_w = crop.shape[:2]
            
            # 의 긴 쪽을 기준으로 640 비율 계산
            scale = 640 / max(curr_h, curr_w)
            new_w = int(curr_w * scale)
            new_h = int(curr_h * scale)
            
            # 리사이즈
            resized_crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            save_name = f"{base_name}_{i}.png"
            save_path = os.path.join(resize_dir, save_name)
            cv2.imwrite(save_path, resized_crop)

        count += 1
        if count % 1000 == 0:
            print(f"현재 {count}장 처리 완료")

print(f"완료: 총 {count}장 처리 완료")
