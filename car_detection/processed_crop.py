import cv2
import os
from tqdm import tqdm

DATA_DIR = '/mnt/hdd_6tb/cjy/processed_full'
SAVE_DIR = '/mnt/hdd_6tb/cjy/processed_carcrop'
TARGET = 640

for split in ['train', 'validation']:
    in_img_dir = os.path.join(DATA_DIR, 'images', split)
    in_lbl_dir = os.path.join(DATA_DIR, 'labels', split)
    
    out_img_dir = os.path.join(SAVE_DIR, 'images', split)
    out_lbl_dir = os.path.join(SAVE_DIR, 'labels', split)
    
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for file_name in tqdm(os.listdir(in_img_dir), desc=f"{split} 진행"):
        # 이미지 읽기
        img = cv2.imread(os.path.join(in_img_dir, file_name))
        if img is None: continue
        H_org, W_org = img.shape[:2]

        # 라벨 읽기
        txt_name = os.path.splitext(file_name)[0] + ".txt"
        txt_path = os.path.join(in_lbl_dir, txt_name)
        if not os.path.exists(txt_path): continue

        with open(txt_path) as f:
            labels = [list(map(float, line.split())) for line in f.readlines()]   # 텍스트 파일을 읽어서 숫자 리스트로 변환

        # 차(1)와 번호판(0) 분리
        cars = [l for l in labels if int(l[0]) == 1]
        plates = [l for l in labels if int(l[0]) == 0]

        for i, (_, cx, cy, cw, ch) in enumerate(cars):
            # 차량 자르기
            x1 = int((cx - cw/2) * W_org)
            y1 = int((cy - ch/2) * H_org)
            x2 = int((cx + cw/2) * W_org)
            y2 = int((cy + ch/2) * H_org)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W_org, x2), min(H_org, y2)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            # 잘린 차량의 크기
            h_c, w_c = crop.shape[:2]

            # B. 긴 쪽을 640에 맞추기
            scale = TARGET / max(h_c, w_c)
            new_w, new_h = int(w_c * scale), int(h_c * scale)
            
            resized = cv2.resize(crop, (new_w, new_h))

            new_txt = []
            for (_, px, py, pw, ph) in plates:
                px_abs, py_abs = px * W_org, py * H_org
                
                # 번호판이 차량 안에 있으면
                if (x1 < px_abs < x2) and (y1 < py_abs < y2):
                    # (절대좌표 - 시작점) / 차량크기
                    nx = (px_abs - x1) / w_c
                    ny = (py_abs - y1) / h_c
                    nw = (pw * W_org) / w_c
                    nh = (ph * H_org) / h_c
                    
                    new_txt.append(f"0 {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")

            base_name = os.path.splitext(file_name)[0] + f"_{i}"
            cv2.imwrite(os.path.join(out_img_dir, base_name + ".png"), resized)
            with open(os.path.join(out_lbl_dir, base_name + ".txt"), 'w') as f:
                f.write('\n'.join(new_txt))

print("완료")