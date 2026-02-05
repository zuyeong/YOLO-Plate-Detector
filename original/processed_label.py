import os
import torch   # pyright: ignore[reportMissingImports]
import glob
import json
from pathlib import Path

device_num = 0
if torch.cuda.is_available():
    torch.cuda.set_device(device_num) 
    device = f"cuda:{device_num}"
else:
    device = "cpu"
print(device)

raw_dir = [
    "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/data/Train/02.라벨링데이터/**/*.json",
    "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/data/Validation/02.라벨링데이터/**/*.json"
]

train_dir = "/home/cjy/workspace/original/processed_original/labels/train"
val_dir = "/home/cjy/workspace/original/processed_original/labels/validation"

count = 0

for path in raw_dir:
    if "Train" in path:
        resize_dir = train_dir
    else:
        resize_dir = val_dir
        
    for json_file in glob.iglob(path, recursive=True):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # 원본 이미지 해상도 추출
        resolution = data["Raw_Data_Info"]["resolution"]
        len_x, len_y = map(int, resolution.split(","))  # "1920, 1080"  -> 1920, 1080 -> img_w = 1920, img_h = 1080
        annotations = data["Learning_Data_Info"].get("annotations", [])
        lines = []

        # boundind box 좌표 추출 
        for ann in annotations:
            for lp in ann.get("license_plate", []):
                bbox = lp.get("bbox", [])
                if len(bbox) != 4:
                    continue
                x, y, w, h = bbox
                
                # normalize
                x_center = (x + w / 2) / len_x
                y_center = (y + h / 2) / len_y
                norm_w = w / len_x
                norm_h = h / len_y
                cls = 0 
                lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        # txt 파일로 저장
        txt_name = Path(json_file).stem + ".txt"
        out_path = os.path.join(resize_dir, txt_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            
        count += 1
        if count % 1000 == 0:
            print(f"현재 {count}개 처리 완료")

print(f"최종 완료: 총 {count}개 처리됨")