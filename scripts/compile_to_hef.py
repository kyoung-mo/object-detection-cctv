#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import shutil
import subprocess


def run_hailomz_compile(model_name: str, hw_arch: str, har_path: Path | None):
    cmd = [
        "hailomz",
        "compile",
        model_name,
        "--hw-arch",
        hw_arch,
    ]

    # 이미 만든 HAR에서 시작하고 싶으면 --har 사용 (Model Zoo User Guide 기준)
    if har_path is not None:
        cmd += ["--har", str(har_path)]

    print("[INFO] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[INFO] hailomz compile finished.")


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
        "--har",
        type=Path,
        default=Path("models/yolov8_person.har"),
        help="HAR file to compile from (optional but recommended)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="yolov8n",
        help="Model Zoo model name (same as optimize step)",
    )
    parser.add_argument(
        "--hw-arch",
        type=str,
        default="hailo8",
        help="target hw arch (hailo8, hailo8l, ...)",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path("."),
        help="root to search for generated .hef",
    )
    parser.add_argument(
        "--output-hef",
        type=Path,
        default=Path("models/yolov8_person.hef"),
        help="final HEF path to copy to",
    )
    args = parser.parse_args()

    har_path = args.har if args.har.exists() else None
    if har_path is None:
        print("[WARN] HAR not found, compiling from previous stage products if possible.")

    run_hailomz_compile(args.model-name, args.hw_arch, har_path)

    # 가장 최근 .hef 찾기
    print(f"[INFO] Searching for latest .hef under: {args.search_root.resolve()}")
    hef_path = find_latest_with_suffix(args.search_root, ".hef")
    if hef_path is None:
        raise RuntimeError("No .hef file found after compilation. "
                           "Check hailomz output directory.")

    print(f"[INFO] Latest HEF found: {hef_path}")

    # models/로 복사
    args.output_hef.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(hef_path, args.output_hef)
    print(f"[INFO] HEF copied to: {args.output_hef.resolve()}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
