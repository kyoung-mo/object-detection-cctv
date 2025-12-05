#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time


def run_hailomz_optimize(
    model_name: str,
    onnx_path: Path,
    calib_dir: Path,
    hw_arch: str,
    classes: int,
):
    cmd = [
        "hailomz",
        "optimize",
        model_name,
        "--hw-arch",
        hw_arch,
        "--ckpt",
        str(onnx_path),
        "--calib-path",
        str(calib_dir),
        "--classes",
        str(classes),
    ]

    print("[INFO] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[INFO] hailomz optimize finished.")


def find_latest_with_suffix(root: Path, suffix: str) -> Path | None:
    candidates = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(suffix):
                p = Path(dirpath) / f
                candidates.append(p)

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("models/yolov8_person.onnx"),
        help="input ONNX model path",
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=Path("data/calib_images"),
        help="directory with calibration images",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="yolov8n",
        help="Model Zoo model name (must exist in hailo_model_zoo cfg)",
    )
    parser.add_argument(
        "--hw-arch",
        type=str,
        default="hailo8",
        help="target hw arch (hailo8, hailo8l, ...)",
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=1,
        help="number of classes for post-processing",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path("."),
        help="root directory to search for generated .har",
    )
    parser.add_argument(
        "--output-har",
        type=Path,
        default=Path("models/yolov8_person.har"),
        help="final HAR path to copy to",
    )
    args = parser.parse_args()

    if not args.onnx.exists():
        raise FileNotFoundError(f"ONNX not found: {args.onnx}")
    if not args.calib-dir.exists():
        raise FileNotFoundError(f"Calibration dir not found: {args.calib_dir}")

    # 1) hailomz optimize 실행
    run_hailomz_optimize(
        model_name=args.model_name,
        onnx_path=args.onnx,
        calib_dir=args.calib_dir,
        hw_arch=args.hw_arch,
        classes=args.classes,
    )

    # 2) 가장 최근에 생성된 .har 찾기
    print(f"[INFO] Searching for latest .har under: {args.search_root.resolve()}")
    har_path = find_latest_with_suffix(args.search_root, ".har")
    if har_path is None:
        raise RuntimeError("No .har file found after optimization. "
                           "Check hailomz output directory.")

    print(f"[INFO] Latest HAR found: {har_path}")

    # 3) models/로 복사
    args.output_har.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(har_path, args.output_har)
    print(f"[INFO] HAR copied to: {args.output_har.resolve()}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
