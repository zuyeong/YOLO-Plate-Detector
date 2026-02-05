from ultralytics import YOLO  # pyright: ignore[reportMissingImports]
from pathlib import Path
import shutil

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
results = model.train(
    data= "/home/cjy/workspace/original/processed_original/data.yaml",
    epochs=100,  # 기본값
    imgsz=640,   # 기본값
    lr0=0.01,     # 기본값 
    batch=-1,    # 자동 모드
    device=0,    # GPU 번호 
    workers=8    
    )

best_pt = Path(results.save_dir) / "weights" / "best.pt"
out_pt = Path("YOLOv11n.pt")
shutil.copy(best_pt, out_pt)