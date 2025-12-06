# scripts/compile_har_to_hef.py

import argparse
import os
from hailo_sdk_client import ClientRunner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--har-path",
        type=str,
        default="hailo/quantized/yolov8_person_quantized.har",
        help="Input quantized HAR path",
    )
    parser.add_argument(
        "--hef-path",
        type=str,
        default="models/yolov8_person.hef",
        help="Output HEF path",
    )
    parser.add_argument(
        "--hw-arch",
        type=str,
        default="hailo8",
        help="Hailo HW arch (hailo8 / hailo8l 등)",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    har_path = args.har_path
    hef_path = args.hef_path

    if not os.path.exists(har_path):
        raise FileNotFoundError(f"HAR not found: {har_path}")

    os.makedirs(os.path.dirname(hef_path), exist_ok=True)

    print(f"[Hailo] HW arch      : {args.hw_arch}")
    print(f"[Hailo] Input  HAR   : {har_path}")
    print(f"[Hailo] Output HEF   : {hef_path}")

    # Hailo SDK 클라이언트 생성
    runner = ClientRunner(hw_arch=args.hw_arch)

    # 양자화된 HAR 로드 (이미 quantized.har 이므로 여기서는 최적화/양자화 안 함)
    print("[Hailo] Loading quantized HAR...")
    hn = runner.load_har(har_path)

    print("[Hailo] Compiling to HEF...")
    # ⚠️ SDK 버전에 따라 API 이름이 다를 수 있음
    # 일반적으로는 compile() -> HEF 객체 반환 후 save() 패턴을 많이 씀
    hef = runner.compile(hn)
    hef.save(hef_path)

    print(f"[Hailo] HEF saved to: {hef_path}")

if __name__ == "__main__":
    main()
