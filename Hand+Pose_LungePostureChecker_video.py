import cv2
import mediapipe as mp
import time
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from PIL import ImageFont, ImageDraw, Image #中文顯示
import numpy as np  # 陣列與數值運算

# ========== 開啟影片 ==========

#cap = cv2.VideoCapture(0)              # 使用預設攝影機（0 表內建或第一個攝影機）
VideoSource="精華弓步蹲_U_638geC90o.mp4" # ← 改為影片來源
cap = cv2.VideoCapture(VideoSource)   
if not cap.isOpened():                # 如果攝影機無法開啟就結束
    print("❌ 無法開啟影片")
    exit()

# 擷取影片資訊
fps = int(cap.get(cv2.CAP_PROP_FPS))
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 準備寫入影片（先預設但不寫入）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 處理副檔名忽略大小寫
base_name = os.path.splitext(VideoSource)[0]  # 去掉副檔名
output_path = f"{base_name}_OutputAnnotated.mp4"

# 若檔案已存在，自動重命名避免覆蓋
if os.path.exists(output_path):
    timestamp = int(time.time())
    output_path = f"{base_name}_OutputAnnotated_{timestamp}.mp4"
    print(f"⚠️ 已存在同名輸出檔，已自動改名為：{output_path}")

# 寫入器初始化
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))


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

# ====== 字型快取（全域）======
_FONT_CACHE = {}
DEFAULT_FONT_PATH = "C:/Windows/Fonts/msjh.ttc"  # 你原本用的路徑
# ========= 中文 + 外框 文字繪製 =========
def draw_chinese_text_with_outline(img, text, position,
    font_path=DEFAULT_FONT_PATH,
    font_size=24,
    text_color=(255,255,255),
    outline_color=(0,0,0),
    bg_color=None,
    padding=4,
    with_outline=True):
    """
    使用 PIL 在 OpenCV 影像上繪製中文文字。
    支援：字型物件快取（避免每次都 truetype 讀檔）、可選擇是否描邊、是否加底色。
    """
    # OpenCV(BGR) → PIL(RGB)
    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        img_pil = img

    draw = ImageDraw.Draw(img_pil)

    # ===== 字型快取 =====
    key = (font_path, font_size)
    font = _FONT_CACHE.get(key)
    if font is None:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            # 萬一路徑失敗，退回預設或 PIL 內建字型（英文/數字可用，中文可能空白）
            try:
                font = ImageFont.truetype(DEFAULT_FONT_PATH, font_size)
            except Exception:
                font = ImageFont.load_default()
        _FONT_CACHE[key] = font  # 放入快取

    # 量測文字尺寸
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    x, y = position

    # 底色
    if bg_color is not None:
        draw.rectangle([x - padding, y - padding, x + text_w + padding, y + text_h + padding], fill=bg_color)

    # 外框
    if with_outline:
        # 8 個方向 + 十字，描邊更實心一些
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)

    # 正文
    draw.text((x, y), text, font=font, fill=text_color)

    # PIL(RGB) → OpenCV(BGR)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ========= 姿勢分類（弓部蹲，左右分開差距） =========
def _diff_to_range(angle, lo=80.0, hi=100.0):
    """回傳角度相對於[lo, hi]的差距；在區間內則為0"""
    if angle < lo:
        return lo - angle
    if angle > hi:
        return angle - hi
    return 0.0
# ========= 姿勢分類弓部蹲 =========
def classify_pose(person):
    """
    規則（弓部蹲）：
    - 左膝與右膝角度同時在 80~100 度 → 「弓部蹲-良好」
    - 否則，只要任一 <120° → 「弓部蹲-不良」，並回傳左右膝分別的差距
    - 其他 → 「站著」
    """
    try:
        LEFT_HIP, RIGHT_HIP = 23, 24
        LEFT_KNEE, RIGHT_KNEE = 25, 26
        LEFT_ANKLE, RIGHT_ANKLE = 27, 28

        left_knee_angle  = calculate_angle(person[LEFT_HIP],  person[LEFT_KNEE],  person[LEFT_ANKLE])
        right_knee_angle = calculate_angle(person[RIGHT_HIP], person[RIGHT_KNEE], person[RIGHT_ANKLE])

        in_left  = 80.0 <= left_knee_angle  <= 100.0
        in_right = 80.0 <= right_knee_angle <= 100.0

        if in_left and in_right:
            return "弓部蹲-良好", None, None

        if (left_knee_angle < 120.0) or (right_knee_angle < 120.0):
            diff_left  = _diff_to_range(left_knee_angle,  80.0, 100.0)
            diff_right = _diff_to_range(right_knee_angle, 80.0, 100.0)
            return "弓部蹲-不良", diff_left, diff_right

        return "站著", None, None

    except Exception:
        return "偵測中...", None, None


# ========= 主迴圈 =========
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    timestamp = int(time.time() * 1000)
    h, w, _ = frame.shape

    # 手部偵測（可選保留）
    hand_result = hand_detector.detect_for_video(mp_image, timestamp)
    if hand_result.hand_landmarks:
        for idx, hand in enumerate(hand_result.hand_landmarks):
            point_color, line_color = HAND_COLORS[idx % len(HAND_COLORS)]
            for i, lm in enumerate(hand):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, point_color, -1)
                cv2.putText(frame, f"{i}", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                #cv2.putText(frame, f"{lm.x:.2f},{lm.y:.2f}", (cx + 5, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            for start, end in HAND_CONNECTIONS:
                x1, y1 = int(hand[start].x * w), int(hand[start].y * h)
                x2, y2 = int(hand[end].x * w), int(hand[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)

    # ===== 姿勢偵測 =====
    pose_result = pose_detector.detect_for_video(mp_image, timestamp)
    if pose_result.pose_landmarks:
        for idx, person in enumerate(pose_result.pose_landmarks):
            point_color, line_color = PERSON_COLORS[idx % len(PERSON_COLORS)]
            px, py = int(person[0].x * w), int(person[0].y * h)

            # ===== 分類姿勢並決定顏色 =====
            pose_label, diff_left, diff_right = classify_pose(person)
            if "不良" in pose_label:
                display_color = (255, 0, 0)
            elif "良好" in pose_label:
                display_color = (0, 255, 0)
            else:
                display_color = (0,255,255) #跟隨人骨架色用point_color

            for i, lm in enumerate(person):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, point_color, -1)
                cv2.putText(frame, f"{i}", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                cv2.putText(frame, f"{lm.x:.2f},{lm.y:.2f}", (cx + 5, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            for start, end in POSE_CONNECTIONS:
                x1, y1 = int(person[start].x * w), int(person[start].y * h)
                x2, y2 = int(person[end].x * w), int(person[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)

            # 姿勢標籤（中文）顯示
            frame = draw_chinese_text_with_outline(
                frame, f"姿勢：{pose_label}", (px, max(py - 70, 0)),
                font_path="C:/Windows/Fonts/msjh.ttc",
                font_size=24, text_color=display_color, outline_color=(0,0,0),
                with_outline=True
            )

            # 顯示差距
            if pose_label == "弓部蹲-不良":
                if diff_left is not None and diff_left > 0:
                    frame = draw_chinese_text_with_outline(
                        frame, f"左膝距標準差 {diff_left:.0f}°",
                        position=(px, max(py - 20, 20)),
                        font_path="C:/Windows/Fonts/msjh.ttc", font_size=22,
                        text_color=(255, 255, 255), outline_color=(0, 0, 0),
                        with_outline=True
                    )
                if diff_right is not None and diff_right > 0:
                    frame = draw_chinese_text_with_outline(
                        frame, f"右膝距標準差 {diff_right:.0f}°",
                        position=(px, max(py - 40, 40)),
                        font_path="C:/Windows/Fonts/msjh.ttc", font_size=22,
                        text_color=(255, 255, 255), outline_color=(0, 0, 0),
                        with_outline=True
                    )

            # 左右膝蓋角度顯示
            try:
                left_angle = calculate_angle(person[23], person[25], person[27])
                x25, y25 = int(person[25].x * w), int(person[25].y * h)
                frame = draw_chinese_text_with_outline(
                    frame, f"左膝：{left_angle:.0f}°", (x25 - 50, y25 - 40),
                    font_size=24, text_color=(255,255,255), outline_color=(0,0,0)
                )

                right_angle = calculate_angle(person[24], person[26], person[28])
                x26, y26 = int(person[26].x * w), int(person[26].y * h)
                frame = draw_chinese_text_with_outline(
                    frame, f"右膝：{right_angle:.0f}°", (x26 - 50, y26 - 40),
                    font_size=24, text_color=(255,255,255), outline_color=(0,0,0)
                )
            except:
                pass

    else:
        cv2.putText(frame, "No person detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 顯示畫面
    cv2.imshow("Pose + Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # 寫入目前這一幀
    out.write(frame)

# ========= 結束處理 =========
cap.release()
cv2.destroyAllWindows()

# 詢問使用者是否要保留儲存影片
save = input("是否要儲存結果影片？(y/n): ").strip().lower()
if save not in ['y', 'yes'] :
    os.remove(output_path)
    print("❌ 已刪除影片")
else:
    print("✅ 影片已儲存於：", output_path)

