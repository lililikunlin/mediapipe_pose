import cv2
import mediapipe as mp
import time
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from PIL import ImageFont, ImageDraw, Image #中文顯示
import numpy as np  # 陣列與數值運算

# ========= 模型載入：Pose =========
pose_model_path = 'pose_landmarker_full.task'
pose_base = python.BaseOptions(model_asset_path=pose_model_path)
pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base,
    running_mode=vision.RunningMode.VIDEO,
    output_segmentation_masks=False,
    num_poses=3,
    min_pose_detection_confidence=0.6,
    min_pose_presence_confidence=0.6,
    min_tracking_confidence=0.5
)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

# ========= 模型載入：Hand =========
hand_model_path = 'hand_landmarker.task'
hand_base = python.BaseOptions(model_asset_path=hand_model_path)
hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# ========= Connection 定義 =========
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (16, 18), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32)
]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20)
]
PERSON_COLORS = [
    ((0, 255, 0), (0, 255, 255)),
    ((255, 0, 0), (255, 255, 0)),
    ((0, 0, 255), (255, 0, 255))
]
HAND_COLORS = [
    ((0, 255, 0), (0, 255, 255)),
    ((255, 0, 0), (255, 255, 0)),
    ((0, 0, 255), (255, 0, 255))
]

# ========= 計算角度函數 =========
def calculate_angle(a, b, c):
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    cx, cy = c.x, c.y

    ab = [ax - bx, ay - by]
    cb = [cx - bx, cy - by]

    dot_product = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.hypot(ab[0], ab[1])
    mag_cb = math.hypot(cb[0], cb[1])

    if mag_ab == 0 or mag_cb == 0:
        return 180.0

    angle_rad = math.acos(dot_product / (mag_ab * mag_cb))
    return math.degrees(angle_rad)

# ========= 姿勢分類函數（加上好壞蹲姿） =========
def classify_pose(person):
    NOSE = 0
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    try:
        l_wrist_y = person[LEFT_WRIST].y
        r_wrist_y = person[RIGHT_WRIST].y
        nose_y = person[NOSE].y
        avg_wrist_y = (l_wrist_y + r_wrist_y) / 2

        if avg_wrist_y < nose_y:
            return "Hands Up"

        left_knee_angle = calculate_angle(person[LEFT_HIP], person[LEFT_KNEE], person[LEFT_ANKLE])
        right_knee_angle = calculate_angle(person[RIGHT_HIP], person[RIGHT_KNEE], person[RIGHT_ANKLE])
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

        if avg_knee_angle < 100:
            if 70 <= avg_knee_angle <= 100:
                return "Squatting-Good"
            else:
                return "Squatting-Bad"
        else:
            return "Standing"
    except:
        return "Detecting..."
    
def draw_chinese_text_with_outline(img, text, position, font_path="C:/Windows/Fonts/msjh.ttc",
                                   font_size=24, text_color=(255,255,255), outline_color=(0,0,0),
                                   bg_color=None, padding=4):
    """
    繪製帶外框的中文文字，可加背景。
    - text_color: 文字主顏色
    - outline_color: 外框色（描邊）
    - bg_color: 若要填背景色（例如黑底白字），否則設 None
    """
    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        img_pil = img

    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)

    # 計算文字大小
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    x, y = position

    # 畫底色框（如需）
    if bg_color is not None:
        draw.rectangle([x - padding, y - padding, x + text_w + padding, y + text_h + padding], fill=bg_color)

    # 畫外框（在主文字四周畫多次偏移）
    for dx in [-1, 1, 0, 0, -1, -1, 1, 1]:  # 8個方向
        for dy in [-1, 1, 0, 0, -1, 1, -1, 1]:
            draw.text((x + dx, y + dy), text, font=font, fill=outline_color)

    # 畫主文字
    draw.text((x, y), text, font=font, fill=text_color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ========= 開啟攝影機 =========
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera open failed")
    exit()

# ========= 主迴圈 =========
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    timestamp = int(time.time() * 1000)
    h, w, _ = frame.shape

    # ===== 姿勢偵測 =====
    pose_result = pose_detector.detect_for_video(mp_image, timestamp)
    if pose_result.pose_landmarks:
        for idx, person in enumerate(pose_result.pose_landmarks):
            point_color, line_color = PERSON_COLORS[idx % len(PERSON_COLORS)]
            px, py = int(person[0].x * w), int(person[0].y * h)

            # ===== 分類姿勢並決定顏色 =====
            pose_label = classify_pose(person)
            if "Good" in pose_label:
                display_color = (0, 255, 0)  # 綠色標準
            elif "Bad" in pose_label:
                display_color = (0, 0, 255)  # 紅色不標準
            else:
                display_color = point_color  # 原本顏色

            cv2.putText(frame, f"Pose: {pose_label}", (px, py - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, display_color, 2)

            # ===== 繪製姿勢點與連線 =====
            cv2.putText(frame, f"Person {idx+1}", (px, py - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, point_color, 2)
            try:
                # 左膝角度（點 23-25-27）
                left_angle = calculate_angle(person[23], person[25], person[27])
                x25, y25 = int(person[25].x * w), int(person[25].y * h)
                frame = draw_chinese_text_with_outline(frame, f"左膝：{left_angle:.0f}°", (x25 - 50, y25 - 40),
                    font_path="C:/Windows/Fonts/msjh.ttc", font_size=24, text_color=(255,255,255),
                    outline_color=(0,0,0),bg_color=None)

                # 右膝角度（點 24-26-28）
                right_angle = calculate_angle(person[24], person[26], person[28])
                x26, y26 = int(person[26].x * w), int(person[26].y * h)
                frame = draw_chinese_text_with_outline(frame, f"右膝：{right_angle:.0f}°", (x26 - 50, y26 - 40),
                    font_path="C:/Windows/Fonts/msjh.ttc", font_size=24, text_color=(255,255,255),
                    outline_color=(0,0,0),bg_color=None)
            except:
                pass

            for i, lm in enumerate(person):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, point_color, -1)
                cv2.putText(frame, f"{i}", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                cv2.putText(frame, f"{lm.x:.2f},{lm.y:.2f}", (cx + 5, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            for start, end in POSE_CONNECTIONS:
                x1, y1 = int(person[start].x * w), int(person[start].y * h)
                x2, y2 = int(person[end].x * w), int(person[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)
    else:
        cv2.putText(frame, "No person detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ===== 手部偵測 =====
    hand_result = hand_detector.detect_for_video(mp_image, timestamp)
    if hand_result.hand_landmarks:
        for idx, hand in enumerate(hand_result.hand_landmarks):
            point_color, line_color = HAND_COLORS[idx % len(HAND_COLORS)]
            for i, lm in enumerate(hand):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, point_color, -1)
                cv2.putText(frame, f"{i}", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                cv2.putText(frame, f"{lm.x:.2f},{lm.y:.2f}", (cx + 5, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            for start, end in HAND_CONNECTIONS:
                x1, y1 = int(hand[start].x * w), int(hand[start].y * h)
                x2, y2 = int(hand[end].x * w), int(hand[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)

    # 顯示畫面
    cv2.imshow("Pose + Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break