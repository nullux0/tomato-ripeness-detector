import cv2
import numpy as np
 
CLASS_NAMES = {0: 'fully_ripe', 1: 'half_ripe', 2: 'unripe'}
COLOR_MAP = {
    0: (0, 0, 255),    # fully_ripe: 빨간색
    1: (0, 165, 255),  # half_ripe: 주황색
    2: (0, 255, 0)     # unripe: 초록색
}
 
 
def draw_boxes(image, boxes, class_ids, confidences):
    img_drawn = image.copy()
    if len(boxes) == 0:
        return img_drawn
 
    for box, cls_id, conf in zip(boxes, class_ids, confidences):
        x1, y1, x2, y2 = map(int, np.round(box))
        color = COLOR_MAP.get(int(cls_id), (255, 255, 255))
        label = f"{CLASS_NAMES.get(int(cls_id), 'Unknown')} {conf:.2f}"
 
        cv2.rectangle(img_drawn, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_drawn, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
 
    return img_drawn
 