import os
import glob
import json
from pathlib import Path

raw_dir = [
    "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/data/Train/02.라벨링데이터/**/*.json",
    "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/data/Validation/02.라벨링데이터/**/*.json"
]

train_dir = "/mnt/hdd_6tb/cjy/processed_quarter/labels/train"
val_dir = "/mnt/hdd_6tb/cjy/processed_quarter/labels/validation"

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
        len_w, len_h = map(int, resolution.split(",")) 
        mid_w, mid_h = len_w // 2, len_h // 2
        
        # 각 구역별 좌표
        crops_info = [
            (0, 0, mid_w, mid_h),           
            (mid_w, 0, len_w - mid_w, mid_h),
            (0, mid_h, mid_w, len_h - mid_h), 
            (mid_w, mid_h, len_w - mid_w, len_h - mid_h) 
        ]
        
        annotations = data["Learning_Data_Info"].get("annotations", [])
        base_name = Path(json_file).stem

        # 4개의 결과 파일 준비
        crop_lines = [[] for _ in range(4)]

        for ann in annotations:
            for lp in ann.get("license_plate", []):
                bbox = lp.get("bbox", [])
                if len(bbox) != 4: continue
                x, y, w, h = bbox
                
                # 중심점
                cx, cy = x + w / 2, y + h / 2
                
                # 중심점이 어느 구역에 속하는지
                for i, (ox, oy, cw, ch) in enumerate(crops_info):
                    if ox <= cx < ox + cw and oy <= cy < oy + ch:
                        # 해당 구역에서의 상대 좌표로 변환
                        nx = cx - ox
                        ny = cy - oy
                        
                        # 해당 구역의 크기로 Normalize
                        norm_cx = nx / cw
                        norm_cy = ny / ch
                        norm_w = w / cw
                        norm_h = h / ch
                        
                        crop_lines[i].append(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}")
                        break # 하나 했으면 끝 

        # 4개의 파일 각각 저장
        for i in range(4):
            txt_name = f"{base_name}_{i}.txt"
            out_path = os.path.join(resize_dir, txt_name)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(crop_lines[i]))

        count += 1
        if count % 1000 == 0:
            print(f"현재 {count}개 처리 완료")

print(f"최종 완료: 총 {count}개 원본 처리됨")