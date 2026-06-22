import streamlit as st
import cv2
import numpy as np
import traceback
import tempfile
import os
import time
 
from inference import load_model, predict
from utils import draw_boxes
 
# 설정:하드코딩 분리
MODEL_PATH = "best.onnx"
VIDEO_TYPES = ['mp4', 'avi', 'mov']
 
st.set_page_config(page_title="토마토 숙도 탐지기", page_icon="🍅", layout="wide")
model = load_model(MODEL_PATH)
 
#사이드바: 실시간 제어
st.sidebar.header("⚙️ 설정 (Settings)")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.00, 1.00, 0.40, 0.05,
    help="오분류(FP)를 줄이려면 임계값을 높이세요")
iou_threshold = st.sidebar.slider(
    "IoU Threshold (NMS)", 0.0, 1.0, 0.45, 0.05)
frame_skip = st.sidebar.slider(
    "프레임 스킵", 1, 10, 2)
capture_width = st.sidebar.select_slider(
    "영상 처리 해상도 (가로 기준)", options=[480, 640, 854, 1280, 1920], value=640,
    help="원본 프레임을 줄여서 전처리/그리기/화면 전송 비용을 줄임"
)
st.sidebar.markdown("---")
 
# 메인
st.title("🍅 토마토 숙도 탐지 시스템")
st.markdown("과수원 환경에서 수확이 임박한 토마토를 탐지하고 숙도를 분류합니다.")
st.caption(f"모델 입력 크기: {model['input_size'][0]}x{model['input_size'][1]} · ONNX Runtime (CPU)")
 
uploaded_video = st.file_uploader("테스트할 동영상을 업로드하세요 (mp4 권장)", type=VIDEO_TYPES)
 
try:
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()
 
        cap = cv2.VideoCapture(tfile.name)
 
        st.markdown("### 📡 프레임별 탐지 결과")
        st.caption("※ 트래킹 미적용 — 누적이 아닌 '현재 프레임' 기준 카운트입니다")
 
        stframe = st.empty()
 
        m1, m2, m3, m4 = st.columns(4)
        metric_ripe = m1.empty()
        metric_half = m2.empty()
        metric_unripe = m3.empty()
        fps_metric = m4.empty()
 
        frame_count = 0
        last_boxes = np.empty((0, 4))
        last_confidences = np.array([])
        last_class_ids = np.array([])
 
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
 
            frame_count += 1
            t0 = time.time()
 
            # 다운스케일
            h0, w0 = frame.shape[:2]
            if w0 > capture_width:
                scale = capture_width / w0
                frame = cv2.resize(frame, (capture_width, int(h0 * scale)))
 
            # frame_skip
            if frame_count == 1 or frame_count % frame_skip == 0:
                last_boxes, last_confidences, last_class_ids = predict(
                    model, frame, conf_threshold, iou_threshold
                )
 
            result_frame_bgr = draw_boxes(frame, last_boxes, last_class_ids, last_confidences)
            result_frame_rgb = cv2.cvtColor(result_frame_bgr, cv2.COLOR_BGR2RGB)
 
            stframe.image(result_frame_rgb, use_container_width=True)
 
            ripe = sum(1 for c in last_class_ids if int(c) == 0)
            half = sum(1 for c in last_class_ids if int(c) == 1)
            unripe = sum(1 for c in last_class_ids if int(c) == 2)
            metric_ripe.metric("🔴 완숙 (현재 프레임)", f"{ripe}개")
            metric_half.metric("🟠 반숙 (현재 프레임)", f"{half}개")
            metric_unripe.metric("🟢 미숙 (현재 프레임)", f"{unripe}개")
 
            elapsed = time.time() - t0
            current_fps = (1 / elapsed) if elapsed > 0 else 0
            fps_metric.metric("⏱️ 처리 속도", f"{current_fps:.1f} FPS")
 
        cap.release()
        os.remove(tfile.name)
        st.success("🎥 영상 처리가 완료되었습니다!")
 
except Exception as e:
    st.error(f"🚨 에러가 발생했습니다: {e}")
    st.code(traceback.format_exc())