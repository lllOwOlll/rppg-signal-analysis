**영상 기반 rPPG 신호 분석 시스템**
: 웹캠 영상에서 얼굴 랜드마크 기반 ROI를 추출하고, RGB 시계열 신호의 전처리 및 주파수 분석을 통해 비접촉 심박수 추정 과정을 구현·검증

주요 기술 스택

Language: Python

Computer Vision / Face Tracking: OpenCV, MediaPipe Face Landmarker

Data Processing: NumPy

Signal Processing / Analysis: SciPy

Data Collection / Analysis: Python time, CSV


주요 개발 내용

MediaPipe 기반 얼굴 랜드마크 검출

- MediaPipe Face Landmarker 모델을 활용하여 실시간 얼굴 랜드마크 검출
- 웹캠 프레임을 MediaPipe 입력 형식으로 변환하여 프레임별 얼굴 위치 추적


Landmark 기반 rPPG ROI 설계

- 얼굴 랜드마크 좌표를 실제 영상의 픽셀 좌표로 변환
- rPPG 신호 측정을 위한 이마, 왼쪽 볼, 오른쪽 볼 영역 선정
- 고정된 사각형 ROI 대신 얼굴 랜드마크 기반 Polygon ROI를 구성하여 얼굴 움직임에 따라 ROI가 추적되도록 구현


ROI Mask 생성 및 피부 영역 분리

- NumPy를 이용하여 영상 크기의 Mask 생성
- OpenCV fillPoly()를 이용해 선택한 얼굴 영역을 Polygon Mask로 생성
- 배경 및 눈·코·입 등의 불필요한 영역을 제외하고 rPPG 분석에 사용할 피부 영역만 선택


ROI RGB 신호 추출

- Mask가 적용된 피부 영역의 R/G/B 채널 평균값을 프레임별로 계산
- time.perf_counter() 기반 실제 Timestamp를 함께 기록
- 추출한 timestamp, R, G, B 데이터를 CSV로 저장하여 후속 rPPG 신호 분석에 활용


ROI RGB 시계열 데이터 수집

- 프레임마다 이마·양 볼 영역의 평균 RGB 값 추출
- 실제 Timestamp와 함께 CSV 저장


신호 전처리

- Green Channel을 rPPG 분석 신호로 사용
- Detrend로 장기적인 밝기 변화 제거
- Band-pass Filter (0.75 ~ 3.0 Hz)로 심박 범위인 45 ~ 180 BPM에 해당하는 주파수 성분 추출


주파수 분석

- Welch PSD를 이용해 주파수별 Power 분석
- find_peaks()를 이용해 주요 Peak 검출


BPM 추정 및 검증

- 검출 주파수에 × 60하여 BPM으로 변환
- 실제 측정 심박수와 rPPG 결과 비교
- 실제 심박에 가까운 Peak가 존재하지만 최대 Power Peak가 실제 심박과 일치하지 않는 문제 발견
- 사람 없는 환경을 추가 측정하여 환경 노이즈와 비교


결과 및 검증

- MediaPipe Face Landmarker를 활용하여 얼굴 움직임에 따라 추적되는 이마·양 볼 ROI 구현
- ROI 영역의 RGB 평균값을 프레임 단위 시계열 데이터로 추출
- Green Channel 신호에 Detrend 및 Band-pass Filter를 적용하여 심박 주파수 대역 분석
- Welch PSD를 이용하여 주요 주파수 Peak를 검출하고 BPM으로 변환
- 실제 심박수와 비교 측정을 통해 rPPG 신호 처리 과정을 검증


한계 및 개선 방향

- 단순 PSD 최대 Peak 기반 심박 추정에서는 실제 심박수와 오차가 발생
- 사람 없는 환경에서도 특정 주파수 Peak가 검출되어 조명 및 환경 변화가 주기 신호로 인식될 가능성 확인
- 향후 RGB 기반 BVP 추출, 신호 품질 평가 및 움직임·조명 변화 보정 등을 통한 정확도 개선 필요
