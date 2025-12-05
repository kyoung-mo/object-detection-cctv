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
- 임베디드 CCTV 형태로 쉽게 재사용 가능한 **GPIO 제어(LED, 부저 등) + 모드 구성** 제공

---

## 2. 시스템 구성

### 2.1 하드웨어

- Raspberry Pi 5
- Hailo-8 AI 가속기
- Raspberry Pi 카메라 모듈 또는 USB 카메라
- LED, 부저, 진동 모듈 등 (GPIO 제어용, 선택사항)

### 2.2 소프트웨어

- Python 3.x
- Ultralytics YOLOv8 (학습 / ONNX export)
- ONNXRuntime (CPU 추론용)
- Hailo Dataflow Compiler, Hailo Model Zoo, HailoRT (HAR/HEF 변환 및 추론)
- OpenCV 등 시각화/카메라 처리 라이브러리

---

## 3. 디렉토리 구조

프로젝트 전체 구조는 다음과 같이 구성된다.

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
│  ├─ visualization.py     # 바운딩 박스, 라벨, FPS 오버레이
│  ├─ gpio_control.py      # LED, 부저, 진동 모듈 제어 (옵션)
│  ├─ modes/
│  │  ├─ live_cctv_mode.py # 실시간 CCTV 모니터링 모드
│  │  └─ record_mode.py    # 녹화/로그 중심 모드 (필요 시)
│  └─ utils/
│     ├─ config_loader.py  # config.yaml, paths.yaml 로드
│     ├─ logging_utils.py  # 로그 포맷/핸들러 설정
│     ├─ timer.py          # FPS, latency 측정 유틸
│     └─ hailo_utils.py    # Hailo 장치 초기화, HEF 로드, 스트림 설정
└─ scripts/
   ├─ export_to_onnx.py        # models/best.pt → models/yolov8_person.onnx
   ├─ convert_to_har.py        # ONNX → HAR (hailomz optimize 래퍼)
   ├─ compile_to_hef.py        # HAR → HEF (hailomz compile 래퍼)
   ├─ run_benchmark_cpu.sh     # CPU 환경 성능 측정 스크립트
   ├─ run_benchmark_hailo.sh   # Hailo 환경 성능 측정 스크립트
   └─ profile_fps_latency.py   # 두 환경 성능 비교 및 시각화
```

---

## 4. 디렉토리/파일별 역할

### 4.1 `config/`

- `config.yaml`  
  - 앱의 전반적인 동작 모드를 정의하는 설정 파일이다.  
  - 예: 사용 모드(live/record), Hailo 사용 여부, 카메라 해상도, confidence / IoU threshold, GPIO 사용 여부 등.

- `paths.yaml`  
  - 모델, 데이터, 로그 등의 경로를 한 곳에서 관리한다.  
  - 코드에서는 경로를 직접 하드코딩하지 않고, 이 파일을 통해 참조하도록 구성한다.

### 4.2 `models/`

- YOLOv8 학습 결과(`best.pt`)와, 이를 기반으로 변환한  
  ONNX / HAR / HEF 파일을 한 디렉토리에서 관리한다.
- 변환 파이프라인은 `scripts/` 의 스크립트들을 통해 수행하도록 설계한다.

### 4.3 `data/`

- `calib_images/` : Hailo 양자화(PTQ) 시 사용될 calibration 이미지.
- `sample_videos/` : 성능 비교/데모에 사용할 테스트 영상.
- `results/` : 성능 평가 결과 그래프, 예시 이미지 등을 저장.

### 4.4 `logs/`

- `app.log` : 실시간 실행 중 발생하는 정보/경고/에러 로그.
- `benchmark.csv` : CPU vs Hailo 성능 측정 결과를 CSV 형식으로 기록하여,  
  이후 그래프 작성이나 논문/보고서 작성 시 재활용할 수 있도록 한다.

### 4.5 `src/`

애플리케이션의 실제 동작 로직들이 위치한다.

- `main.py`  
  - 설정 파일을 로드하고,  
  - CPU/Hailo 모드 및 실행 모드(live/record 등)를 선택하여  
  - 적절한 모듈을 호출하는 엔트리 포인트.

- `camera.py`  
  - 라즈베리파이 카메라 모듈 또는 USB 웹캠에서 프레임을 읽어오는 래퍼.

- `detection_cpu.py` / `detection_hailo.py`  
  - 각각 ONNXRuntime, HailoRT를 사용하여 추론을 수행한다.  
  - 전처리/후처리는 `postprocess.py`와 공유하여,  
    동일한 로직으로 결과를 비교할 수 있도록 설계한다.

- `postprocess.py`  
  - YOLOv8 출력 텐서를 bounding box, score, class id로 변환하는 공통 로직.
  - NMS, confidence threshold 적용 등을 담당한다.

- `visualization.py`  
  - 프레임 위에 박스, 라벨, FPS 등을 그리는 기능.

- `gpio_control.py`  
  - 특정 클래스(예: person) 탐지 시 LED 켜기, 부저/진동 모듈 작동 등  
    임베디드 CCTV 형태의 피드백을 제공하는 모듈.

- `modes/`  
  - `live_cctv_mode.py` : 실시간 화면 출력 + 알림 중심의 모드.
  - `record_mode.py` : 영상/로그 저장 중심의 모드 (필요 시 추가 구현).

- `utils/`  
  - `config_loader.py` : `config.yaml`, `paths.yaml`을 읽어 Python 객체로 변환.
  - `logging_utils.py` : 콘솔/파일 로그 설정.
  - `timer.py` : FPS, latency 측정에 사용되는 간단한 타이머 유틸.
  - `hailo_utils.py` : Hailo 장치 초기화, HEF 로드, 스트림 생성 등을 담당.

### 4.6 `scripts/`

- 모델 변환 및 성능 측정을 위한 스크립트 모음이다.

- `export_to_onnx.py`  
  - `models/best.pt`를 로드하여 `models/yolov8_person.onnx`로 내보내는 스크립트.

- `convert_to_har.py`  
  - Hailo Model Zoo / Dataflow Compiler를 호출하여 ONNX → HAR 변환을 수행한다.

- `compile_to_hef.py`  
  - HAR 파일을 타깃(Hailo-8)에 맞게 HEF로 컴파일한다.

- `run_benchmark_cpu.sh`, `run_benchmark_hailo.sh`  
  - 동일한 입력 조건에서 CPU와 Hailo 환경 각각의 FPS, latency 등을 측정하고,  
    결과를 `logs/benchmark.csv` 등에 기록하도록 설계한다.

- `profile_fps_latency.py`  
  - 측정된 성능 데이터를 불러와 그래프를 생성하고,  
  - 논문/보고서/발표 자료에 사용할 수 있는 시각화 결과를 만들어낸다.

---

## 5. 향후 계획

- YOLOv8 단일 클래스(person) 모델에 대해  
  ONNX / HAR / HEF 변환 파이프라인을 정식 구현 및 검증
- 라즈베리파이 5 + Hailo-8 환경에서의
  - FPS, latency, mAP, 전력 소모 등 정량 지표 측정
  - CPU 단독 추론과의 성능 비교 실험
- 실시간 골목길 CCTV 시나리오와 연계하여
  - 위험 상황(야간 보행자, 특정 감시 구역 출입 등)에 대한 경고/알림 기능 구현
  - 향후 졸업논문 및 공모전/발표 자료와 연계 가능한 형태로 결과 정리

이 저장소는 위와 같은 목표를 기준으로,  
모델 학습 결과부터 임베디드 추론, 그리고 성능 분석까지의 전체 과정을  
재현 가능하고 체계적으로 관리하는 것을 목표로 한다.
