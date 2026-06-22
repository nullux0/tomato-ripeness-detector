# 🍅 웹 기반 토마토 숙도 탐지 시스템
## 1.개요
프로젝트 소개본 프로젝트는 과수원에서의 토마토 객체 탐지 시스템입니다.
YOLOv8 모델을 기반으로 학습을 진행하였으며, ONNX 형식으로 변환하여 추론 속도를 최적화했습니다.
<img width="1275" height="627" alt="스크린샷 2026-06-22 120604" src="https://github.com/user-attachments/assets/5c1beb40-ce8e-4095-a221-667bd05aecf1" />

<img width="632" height="578" alt="스크린샷 2026-06-22 120629" src="https://github.com/user-attachments/assets/603b1248-0995-466b-a997-4946ac6a1086" />


## 2. 개발 환경 및 의존성 
- OS: Windows / Mac / Linux (Cross-platform)
- Language: Python 3.9+
- Framework: ONNX Runtime, Ultralytics (YOLOv8), Streamlit
- 주요 패키지: requirements.txt 참조

## 3. 상세 설치 및 실행 방법
### 3-1. 
모델 가중치 파일 다운로드GitHub 용량 제한(100MB)으로 인해 학습된 모델 파일은 외부 클라우드에 업로드하였습니다. 아래 링크에서 모델을 다운로드하여 프로젝트 최상위 폴더(app.py와 같은 위치)에 넣어주세요.

[🔗 YOLOv8 ONNX 모델 가중치 다운로드 링크] : https://drive.google.com/file/d/1a5u2VST4r9XnzAPbX3bekNC_FiFmD9CX/view?usp=drive_link 
### 3-2. 
로컬 환경 실행 방법터미널을 열고 아래 명령어를 순서대로 실행합니다.

#### 1. 저장소 클론 (다운로드)
   ```
   git clone https://github.com/nullux0/tomato-ripeness-detector.git

cd tomato-ripeness-detector
   ```

#### 2. 필요 패키지 설치
   ```
    pip install -r requirements.txt
   ```

#### 3. Streamlit 웹 서버 실행
   ```
    streamlit run app.py
   ```


## 4. 데이터 파이프라인 
1. 데이터 수집: Roboflow에서 토마토 숙도별(미숙/착색중/완숙) 이미지 및 Bounding Box 데이터 확보 (박스 약 2,422개)
2. 전처리: 모든 이미지를 640x640 해상도로 Auto-Orient 및 Resize 처리
3. 모델 학습: Google Colab (T4 GPU) 환경에서 YOLOv8n 모델을 활용하여 Transfer Learning 진행
4. 최적화: 추론 지연 시간(Latency) 최소화를 위해 PyTorch 모델(.pt)을 ONNX 런타임 환경으로 변환
5. 서빙: Streamlit을 활용하여 사용자가 영상을 업로드하면 즉시 탐지 결과를 시각화해주는 Web UI 구축


## 5. 팀원 역할 분담
[공소영]:
- 커스텀 데이터셋 수집 (Roboflow)
- YOLOv8 모델 Fine-Tuning 및 평가 (Colab)
- 모델 ONNX 변환 및 최적화
- Streamlit 기반 웹 데모 UI/UX 구현 및 깃허브 배포
- 보고서 작성

[정예린]:
- streamlit 기반 UI(app.py) 구현
