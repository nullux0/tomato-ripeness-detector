import streamlit as st
import numpy as np
import cv2
import os
import onnxruntime as ort
 
# 설정:하드코딩 분리
DEFAULT_MODEL_PATH = "best.onnx"
DEFAULT_INPUT_SIZE = 640   # ONNX export 시 입력 크기가 동적일 때 사용할 기본값
 
 #@st.cache_resource로 모델을 메모리에 한 번만 올리고 재사용 (싱글톤)
@st.cache_resource
def load_model(model_path=DEFAULT_MODEL_PATH):

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = os.cpu_count() 
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
 
    #ONNX Runtime InferenceSession 로드 (CPU 전용)
    session = ort.InferenceSession(
        model_path, sess_options=sess_options, providers=['CPUExecutionProvider']
    )
 
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    shape = input_info.shape 
 
    h = shape[2] if isinstance(shape[2], int) else DEFAULT_INPUT_SIZE
    w = shape[3] if isinstance(shape[3], int) else DEFAULT_INPUT_SIZE
 
    return {
        'session': session,
        'input_name': input_name,
        'input_size': (w, h),
    }
 
 #비율 유지하며 리사이즈 + 패딩 (YOLO 표준 전처리)
def letterbox(image, new_size, color=(114, 114, 114)):
    h0, w0 = image.shape[:2]
    new_w, new_h = new_size
    r = min(new_w / w0, new_h / h0)
    resized_w, resized_h = int(round(w0 * r)), int(round(h0 * r))
 
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
 
    pad_w, pad_h = new_w - resized_w, new_h - resized_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
 
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)
 
 
def preprocess(image_bgr, input_size):
    padded, ratio, pad = letterbox(image_bgr, input_size)
    img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))      # HWC -> CHW
    img_batch = img_chw[np.newaxis, :].copy()        # 배치 차원 추가 [1, C, H, W]
    return img_batch, ratio, pad
 
 #confidence 필터 + NMS 직접 수행
def postprocess(output, conf_threshold, iou_threshold, ratio, pad, orig_shape):
    pred = output[0]
 
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
 
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_scores)), class_ids]
 
    mask = confidences > conf_threshold
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
 
    if len(boxes_xywh) == 0:
        return np.empty((0, 4)), np.array([]), np.array([])
 
    # NMSBoxes는 [x_top_left, y_top_left, w, h] 형식
    nms_boxes = boxes_xywh.copy()
    nms_boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    nms_boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
 
    indices = cv2.dnn.NMSBoxes(
        nms_boxes.tolist(), confidences.tolist(), conf_threshold, iou_threshold
    )
    if len(indices) == 0:
        return np.empty((0, 4)), np.array([]), np.array([])
    indices = np.array(indices).flatten()
 
    boxes_xywh = boxes_xywh[indices]
    confidences = confidences[indices]
    class_ids = class_ids[indices]
 
    # center xywh -> xyxy (모델 입력 좌표계)
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
 
    # 레터박스 패딩 제거 + 원본 좌표로 역변환
    boxes_xyxy[:, [0, 2]] -= pad[0]
    boxes_xyxy[:, [1, 3]] -= pad[1]
    boxes_xyxy /= ratio
 
    h0, w0 = orig_shape
    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, w0)
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, h0)
 
    return boxes_xyxy, confidences, class_ids
 
 
def predict(model, image_bgr, conf_threshold=0.4, iou_threshold=0.45):
    session = model['session']
    input_name = model['input_name']
    input_size = model['input_size']
 
    img_batch, ratio, pad = preprocess(image_bgr, input_size)
    outputs = session.run(None, {input_name: img_batch})
 
    boxes, confidences, class_ids = postprocess(
        outputs[0], conf_threshold, iou_threshold, ratio, pad, image_bgr.shape[:2]
    )
    return boxes, confidences, class_ids