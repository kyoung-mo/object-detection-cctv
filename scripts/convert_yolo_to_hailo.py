#!/usr/bin/env python3
"""
YOLOv8 .pt  ->  .onnx  ->  Hailo .har / .hef 변환 스크립트

사용 예시:

(venv) $ python scripts/convert_yolo_to_hailo.py \
    --weights models/best.pt \
    --net-name yolov8_person \
    --imgsz 640 \
    --onnx-path onnx/best.onnx \
    --hw-arch hailo8 \
    --calib-size 200 \
    --calib-batch-size 8 \
    --calib-dir hailo/calib

※ calib-dir 안에 jpg/png가 없으면 랜덤 캘리브레이션 이미지를 사용합니다.
"""

import argparse
import os
from pathlib import Path
import glob

import numpy as np
import torch
from torch.nn.modules.container import Sequential

# --- Hailo SDK ---
from hailo_sdk_client import ClientRunner

# --- YOLO / Ultralytics ---
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.head import Detect
from ultralytics import YOLO


def add_torch_safe_globals():
    """
    PyTorch 2.6+ 의 weights_only=True 기본값으로 인한
    UnpicklingError를 막기 위해 YOLO 관련 클래스를 safe_globals에 등록.
    """
    from torch.serialization import add_safe_globals

    globals_to_add = [
        Conv,
        Detect,
        DetectionModel,
        Sequential,
    ]

    add_safe_globals(globals_to_add)


def export_to_onnx(weights_path: Path, onnx_path: Path, imgsz: int = 640) -> Path:
    """
    Ultralytics YOLO .pt → ONNX 변환
    """
    print(f"[ONNX] weights: {weights_path}")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # PyTorch UnpicklingError 회피용
    add_torch_safe_globals()

    # YOLO 로드
    model = YOLO(str(weights_path))

    # export() 는 보통 저장된 파일 경로를 반환함
    print(f"[ONNX] Exporting to ONNX (imgsz={imgsz}) ...")
    exported = model.export(
        format="onnx",
        opset=12,
        imgsz=imgsz,
        simplify=True,
        dynamic=False,   # 필요하면 True로 바꿀 수 있음
    )

    # exported 가 문자열 또는 Path일 가능성 있음
    exported_path = Path(exported)

    # 사용자가 원하는 onnx_path와 다르면 복사/이동
    if exported_path.resolve() != onnx_path.resolve():
        # 그냥 동일 경로로 복사
        import shutil
        shutil.copy2(exported_path, onnx_path)
        print(f"[ONNX] Copied exported ONNX: {exported_path} -> {onnx_path}")
        return onnx_path

    print(f"[ONNX] Saved ONNX: {exported_path}")
    return exported_path


def load_calib_from_dir(calib_dir: Path, imgsz: int, calib_size: int) -> np.ndarray:
    """
    calib_dir 안의 jpg/png 이미지를 읽어서 (N, H, W, 3) uint8 캘리브레이션 셋 생성.
    이미지 수가 calib_size 보다 적으면 있는 만큼만 사용.
    """
    import cv2

    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    paths = []
    for e in exts:
        paths.extend(glob.glob(str(calib_dir / e)))
    paths = sorted(paths)

    if len(paths) == 0:
        print(f"[Calib] WARNING: {calib_dir} 에서 이미지 파일을 찾지 못함. 랜덤 캘리브레이션으로 대체합니다.")
        return None

    print(f"[Calib] Found {len(paths)} images in {calib_dir}")
    imgs = []
    for p in paths[:calib_size]:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (imgsz, imgsz))
        imgs.append(img)

    if len(imgs) == 0:
        print(f"[Calib] WARNING: 이미지 로드 실패. 랜덤 캘리브레이션으로 대체합니다.")
        return None

    arr = np.stack(imgs, axis=0).astype(np.uint8)
    print(f"[Calib] Calibration shape from images: {arr.shape}")
    return arr


def build_calib_data(imgsz: int, calib_size: int, calib_dir: Path | None) -> np.ndarray:
    """
    캘리브레이션 데이터 생성:
      1) calib_dir이 있으면 실제 이미지를 사용
      2) 없으면 랜덤 uint8 이미지 사용
    """
    if calib_dir is not None:
        calib = load_calib_from_dir(calib_dir, imgsz, calib_size)
        if calib is not None:
            return calib

    # fallback: 랜덤 데이터
    print(f"[Calib] Using random uint8 calibration data: (N={calib_size}, H={imgsz}, W={imgsz}, C=3)")
    calib = np.random.randint(
        0, 256, size=(calib_size, imgsz, imgsz, 3), dtype=np.uint8
    )
    return calib


def onnx_to_hailo(
    onnx_path: Path,
    net_name: str,
    hw_arch: str,
    calib_data: np.ndarray,
    op_level: int,
    comp_level: int,
    calib_batch_size: int,
    har_path: Path,
    hef_path: Path,
    end_nodes: list[str] | None = None,
):
    """
    ONNX → Hailo HAR / HEF 변환
    """
    print(f"[Hailo] Using ONNX: {onnx_path}")
    print(f"[Hailo] Net name   : {net_name}")
    print(f"[Hailo] HW arch    : {hw_arch}")
    print(f"[Hailo] Calibration : {calib_data.shape}")

    har_path.parent.mkdir(parents=True, exist_ok=True)
    hef_path.parent.mkdir(parents=True, exist_ok=True)

    runner = ClientRunner(hw_arch=hw_arch)

    # YOLOv8 DFL 관련 에러가 나면 end_node_names를 지정해줄 수 있음.
    # 예: end_nodes=["/model.22/Concat_3"]
    if end_nodes:
        hn, npz = runner.translate_onnx_model(
            str(onnx_path),
            net_name,
            end_node_names=end_nodes,
        )
    else:
        hn, npz = runner.translate_onnx_model(
            str(onnx_path),
            net_name,
        )

    print(f"[Hailo] Parsed HN: {hn}")
    print(f"[Hailo] NPZ keys: {list(npz.keys())}")

    runner.save_har(str(har_path.with_suffix(".parsed.har")))
    print(f"[Hailo] Saved parsed HAR: {har_path.with_suffix('.parsed.har')}")

    # 모델 스크립트: 최적화 & 캘리브레이션 설정
    model_script = f"""
model_optimization_flavor(optimization_level={op_level}, compression_level={comp_level})
performance_param(compiler_optimization_level=max)
model_optimization_config(calibration, batch_size={calib_batch_size}, calibset_size={calib_data.shape[0]})
"""
    runner.load_model_script(model_script)

    print("[Hailo] Running optimization (quantization)...")
    runner.optimize(calib_data)

    runner.save_har(str(har_path))
    print(f"[Hailo] Saved quantized HAR: {har_path}")

    print("[Hailo] Compiling to HEF...")
    hef = runner.compile()
    with open(hef_path, "wb") as f:
        f.write(hef)
    print(f"[Hailo] Saved HEF: {hef_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 .pt → .onnx → Hailo .har / .hef 변환 스크립트"
    )

    # 1) YOLO → ONNX
    parser.add_argument(
        "--weights",
        type=str,
        default="models/best.pt",
        help="YOLOv8 PyTorch weights (.pt) 경로",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="ONNX/캘리브레이션에서 사용할 입력 이미지 크기 (정사각형)",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="onnx/best.onnx",
        help="저장할 ONNX 파일 경로",
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="이미 ONNX가 있다고 가정하고 .pt→ONNX 단계를 건너뜀",
    )

    # 2) Hailo 변환
    parser.add_argument(
        "--net-name",
        type=str,
        default="yolov8_person",
        help="Hailo 내부에서 사용할 네트워크 이름",
    )
    parser.add_argument(
        "--hw-arch",
        type=str,
        default="hailo8",
        help="타겟 HW 아키텍처 (예: hailo8)",
    )
    parser.add_argument(
        "--calib-size",
        type=int,
        default=200,
        help="캘리브레이션 샘플 개수",
    )
    parser.add_argument(
        "--calib-batch-size",
        type=int,
        default=8,
        help="Hailo optimize 시 batch_size",
    )
    parser.add_argument(
        "--calib-dir",
        type=str,
        default="",
        help="캘리브레이션용 이미지 디렉토리 (비우면 랜덤 생성)",
    )
    parser.add_argument(
        "--op",
        type=int,
        default=1,
        help="optimization_level",
    )
    parser.add_argument(
        "--comp",
        type=int,
        default=0,
        help="compression_level",
    )

    parser.add_argument(
        "--har-path",
        type=str,
        default="hailo/quantized/yolov8_person_quantized.har",
        help="저장할 HAR 경로",
    )
    parser.add_argument(
        "--hef-path",
        type=str,
        default="hailo/hef/yolov8_person.hef",
        help="저장할 HEF 경로",
    )

    parser.add_argument(
        "--end-node",
        type=str,
        default="",
        help="필요 시 end_node_names 로 쓸 노드 이름 (예: /model.22/Concat_3)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(".").resolve()

    weights_path = (project_root / args.weights).resolve()
    onnx_path = (project_root / args.onnx_path).resolve()
    har_path = (project_root / args.har_path).resolve()
    hef_path = (project_root / args.hef_path).resolve()
    calib_dir = Path(args.calib_dir).resolve() if args.calib_dir else None

    if not weights_path.is_file() and not args.skip_onnx:
        raise FileNotFoundError(f"weights not found: {weights_path}")

    # 1) .pt → .onnx
    if args.skip_onnx:
        print(f"[Main] Skipping ONNX export. Using existing: {onnx_path}")
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    else:
        onnx_path = export_to_onnx(weights_path, onnx_path, imgsz=args.imgsz)

    # 2) 캘리브레이션 데이터 준비
    calib_data = build_calib_data(args.imgsz, args.calib_size, calib_dir)

    # 3) ONNX → HAR / HEF
    end_nodes = [args.end_node] if args.end_node else None
    onnx_to_hailo(
        onnx_path=onnx_path,
        net_name=args.net_name,
        hw_arch=args.hw_arch,
        calib_data=calib_data,
        op_level=args.op,
        comp_level=args.comp,
        calib_batch_size=args.calib_batch_size,
        har_path=har_path,
        hef_path=hef_path,
        end_nodes=end_nodes,
    )

    print("\n[Done] .pt → .onnx → .har → .hef 변환 완료 ✅")


if __name__ == "__main__":
    main()
