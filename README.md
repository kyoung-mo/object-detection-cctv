# object-detection-cctv  
라즈베리파이 5와 Hailo-8을 이용한 객체 탐지 기반 CCTV 시스템

---

## 1. 프로젝트 개요

`object-detection-cctv`는 **라즈베리파이 5(Raspberry Pi 5)** 와 **Hailo-8 하드웨어 가속기**를 활용하여,  
골목길 등 실환경에서 사용할 수 있는 **실시간 객체 탐지 CCTV 시스템**을 구현하고,  
동일한 YOLOv8 모델에 대해 **CPU 단독 추론 vs Hailo-8 가속 추론**의 성능을 체계적으로 비교·분석하는 것을 목표로 한다.

이 저장소에서는 다음과 같은 내용을 하나의 파이프라인으로 관리하는 것을 지향한다.

- Ultralytics YOLOv8 기반 **단일 클래스(예: person)** 객체 탐지 모델 학습 결과(`best.pt`) 관리  
- `best.pt` → **ONNX** → **HAR** → **HEF** 로 이어지는 변환 파이프라인 정리  
- 라즈베리파이 5에서의 **CPU(ONNXRuntime) 추론**과 **Hailo-8(HailoRT) 추론** 코드 통합 관리  
- 동일한 입력(카메라/영상)에 대해 **FPS, latency, mAP, 전력** 등의 성능 지표를 비교·로그화  
- 실운영 CCTV 형태로 사용할 수 있도록 **GPIO 경고 시스템 + LCD 디스플레이 출력 기능 통합**  
- 상태에 따라 자동으로 LED/부저/LCD 상태를 갱신하는 **임베디드 알림 시스템 구현**

---

## 2. 시스템 구성

### 2.1 하드웨어

- Raspberry Pi 5  
- Hailo‑8 AI Accelerator  
- Raspberry Pi Camera Module 또는 USB 카메라  
- HD44780 계열 문자 LCD (사진 속 파란 백라이트 LCD)  
- LED (Green / Red)  
- 부저(Buzzer)  
- 모드 전환용 스위치  
- 점검용 진동 모듈 등 선택 요소  

### 2.2 소프트웨어

- Python 3.x  
- Ultralytics YOLOv8  
- ONNXRuntime (CPU 추론용)  
- Hailo Dataflow Compiler, Hailo Model Zoo, HailoRT  
- RPi.GPIO 또는 gpiozero  
- OpenCV  
- LCD 제어 라이브러리 (RPLCD 또는 커스텀 드라이버)

---

## 3. 동작 모드 및 GPIO/LCD 시스템

본 시스템은 **두 가지 모드(Safe / Alert)** 로 동작하며, Raspberry Pi 스위치를 통해 즉시 전환할 수 있다.

### 🟢 3.1 안전 모드 (Safe Mode)

- 사람 감지 여부와 무관하게 **GPIO 장치(LED/부저)를 모두 비활성화**
- LCD 1줄 출력:  
  **`Safe Mode Operating...`**
- 화면에는 YOLO 바운딩박스만 시각적으로 표시됨
- 감시 상황이 아닌 “일반 모드”로 사용할 때 적합

---

### 🔴 3.2 경계 모드 (Alert Mode)

스위치로 진입하면 자동으로 경계 동작 수행

#### (1) **사람 미탐지 상태**
- Green LED ON  
- Red LED OFF  
- Buzzer OFF  
- LCD 출력:  
  **`Alert Mode Operating...`**

#### (2) **사람 감지 상태**
- Green LED OFF  
- Red LED ON  
- Buzzer:  
  **5초에 1번, 0.5초 동안 울림 (반복 패턴)**  
- LCD 출력:  
  **`Person Detected!!`**

경계 모드는 실제 골목길 감시/출입 감지 시나리오에 적합하게 설계되었다.

---

## 4. 디렉토리 구조

```text
object-detection-cctv/
├─ README.md
├─ requirements.txt
├─ config/
│  ├─ config.yaml          # 동작 모드, 임계값, 카메라/장치 설정
│  └─ paths.yaml           # 모델, 데이터, 로그 경로 정의
├─ models/
│  ├─ best.pt              # YOLOv8 학습 결과 (시작점)
│  ├─ yolov8_person.onnx   # best.pt → ONNX 변환 결과
│  ├─ yolov8_person.har    # ONNX → Hailo HAR
│  └─ yolov8_person.hef    # HAR → Hailo HEF (라즈베리파이 실행용)
├─ data/
│  ├─ calib_images/        # Hailo 양자화용 calibration 이미지
│  ├─ sample_videos/       # 성능 비교용 샘플 영상
│  └─ results/             # 실험 결과 이미지/그래프 등
├─ logs/
│  ├─ app.log              # 애플리케이션 실행 로그
│  └─ benchmark.csv        # CPU vs Hailo 성능 측정 결과
├─ src/
│  ├─ main.py              # 엔트리 포인트 (모드, 디바이스 선택)
│  ├─ camera.py            # 카메라 입력(라즈베리파이 카메라 / USB)
│  ├─ detection_cpu.py     # ONNXRuntime 기반 CPU 추론 모듈
│  ├─ detection_hailo.py   # HailoRT 기반 Hailo-8 추론 모듈
│  ├─ postprocess.py       # YOLOv8 bbox 디코딩 및 NMS 공통 로직
│  ├─ visualization.py     # 바운딩박스, 라벨, FPS 오버레이
│  ├─ gpio_control.py      # LED, 부저, 진동 모듈 제어
│  ├─ lcd_display.py       # HD44780 문자 LCD 제어 모듈
│  ├─ modes/
│  │  ├─ live_cctv_mode.py # 실시간 CCTV 모니터링 모드
│  │  └─ record_mode.py    # 녹화/로그 중심 모드
│  └─ utils/
│     ├─ config_loader.py  # config.yaml, paths.yaml 로드
│     ├─ logging_utils.py  # 로그 포맷/핸들러 설정
│     ├─ timer.py          # FPS, latency 측정 유틸
│     └─ hailo_utils.py    # Hailo 초기화 / HEF 로드 / 스트림 설정
└─ scripts/
   ├─ export_to_onnx.py        # best.pt → ONNX
   ├─ convert_to_har.py        # ONNX → HAR
   ├─ compile_to_hef.py        # HAR → HEF
   ├─ run_benchmark_cpu.sh     # CPU 성능 측정
   ├─ run_benchmark_hailo.sh   # Hailo 성능 측정
   └─ profile_fps_latency.py   # 성능 그래프 생성
```

---

## 5. 향후 계획

- YOLOv8 단일 클래스(person) 모델에 대한  
  ONNX → HAR → HEF 변환 파이프라인 완성
- CPU vs Hailo‑8 실험 자동화 파이프라인 구축  
- GPIO + LCD 기반 임베디드 경보 시스템 안정화  
- 실제 골목길 환경 실험을 통한 성능 검증  
- 졸업논문 및 공모전 제출용 그래프·지표 생성 자동화

이 저장소는  
**객체 탐지 모델 개발 → 임베디드 배포 → 실시간 감지 → 경보 시스템까지**  
전체 흐름을 하나의 프로젝트로 재현 가능하게 구성한다.
