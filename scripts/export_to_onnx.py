#!/usr/bin/env python3
from pathlib import Path

import torch
import torch.nn as nn
import torch.serialization as ts

# ★ YOLO 내부 클래스들 import
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv

# ★ torch.load(weights_only=True)가 허용할 클래스들을 등록
ts.add_safe_globals([
    DetectionModel,   # YOLO 모델 클래스
    Conv,             # ultralytics.nn.modules.conv.Conv
    nn.Sequential,    # torch.nn.modules.container.Sequential
])

from ultralytics import YOLO


def export_to_onnx(
    weights_path: Path,
    imgsz: int = 640,
    opset: int = 12,
    simplify: bool = True,
) -> Path:
    print(f"[INFO] Loading YOLOv8 weights: {weights_path}")

    # 여기서 torch.load가 호출되는데,
    # 위에서 DetectionModel을 safe_globals에 넣어서 UnpicklingError를 막은 상태야.
    model = YOLO(str(weights_path))

    print("[INFO] Exporting to ONNX...")
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
    )

    onnx_path = Path(onnx_path)
    print(f"[INFO] Exported ONNX: {onnx_path}")
    return onnx_path


def main():
    project_root = Path(__file__).resolve().parents[1]
    weights = project_root / "models" / "best.pt"

    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    onnx_path = export_to_onnx(weights)

    # 필요하면 이름을 고정해서 관리 (예: yolov8_person.onnx)
    fixed_path = onnx_path.with_name("yolov8_person.onnx")
    if onnx_path != fixed_path:
        onnx_path.rename(fixed_path)
        onnx_path = fixed_path

    print(f"[INFO] Final ONNX path: {onnx_path}")


if __name__ == "__main__":
    main()
