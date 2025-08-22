import cv2
import mediapipe as mp
import time
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from PIL import ImageFont, ImageDraw, Image #中文顯示
import numpy as np  # 陣列與數值運算

# ========= 開啟攝影機 =========
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera open failed")
    exit()

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

# ========= 中文 + 外框 文字繪製 =========
def draw_chinese_text_with_outline(img, text, position,
    font_path="C:/Windows/Fonts/msjh.ttc",
    font_size=24,
    text_color=(255,255,255),
    outline_color=(0,0,0),
    bg_color=None,
    padding=4,
    with_outline=True):
    """
    使用 PIL 在 OpenCV 影像上繪製中文文字。
    可選擇是否描邊、是否加底色。
    """
    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        img_pil = img

    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    x, y = position

    if bg_color is not None:
        draw.rectangle([x - padding, y - padding, x + text_w + padding, y + text_h + padding], fill=bg_color)

    if with_outline:
        for dx in [-1, 1, 0, 0, -1, -1, 1, 1]:
            for dy in [-1, 1, 0, 0, -1, 1, -1, 1]:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)

    draw.text((x, y), text, font=font, fill=text_color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ============幫手函式============
def euclid(p, q):
    return math.hypot(p.x - q.x, p.y - q.y)

# =======開合跳姿勢判定（含條件）=======
def classify_jumping_jack(person):
    """
    回傳：
      raw_state: 'open' / 'closed' / 'transit'
      info : dict，包含 arms_up, legs_apart, legs_together, ratio, ankle_dist, hip_width
    規則：
      open   = arms_up 且 legs_apart (ratio > 1.4)
      closed = (not arms_up) 且 legs_together (ratio < 0.7)
      其他   = transit
    """
    NOSE = 0
    L_HIP, R_HIP = 23, 24
    L_ANKLE, R_ANKLE = 27, 28
    L_WRIST, R_WRIST = 15, 16

    nose = person[NOSE]
    lh, rh = person[L_HIP], person[R_HIP]
    la, ra = person[L_ANKLE], person[R_ANKLE]
    lw, rw = person[L_WRIST], person[R_WRIST]

    hip_width = euclid(lh, rh)                   # 髖寬（正規化單位）
    ankle_dist = euclid(la, ra)                  # 腳踝間距（正規化單位）
    ratio = ankle_dist / max(hip_width, 1e-6)    # 腿部開合比

    # 手是否高於頭（取兩手腕平均）
    avg_wrist_y = (lw.y + rw.y) / 2.0
    arms_up = avg_wrist_y < (nose.y - 0.02)      # 容忍一點 margin

    # 腿部條件（具 hysteresis）
    legs_apart = ratio > 1.4
    legs_together = ratio < 0.7

    if arms_up and legs_apart:
        raw_state = "open"
    elif (not arms_up) and legs_together:
        raw_state = "closed"
    else:
        raw_state = "transit"

    return raw_state, {
        "arms_up": arms_up,
        "legs_apart": legs_apart,
        "legs_together": legs_together,
        "ratio": ratio,
        "ankle_dist": ankle_dist,
        "hip_width": hip_width
    }

# ==========穩定狀態與「過渡」納入計數的有限狀態機==========
STABLE_FRAMES = {"open": 3, "closed": 3, "transit": 2}  # 各狀態需連續多少幀才視為「穩定」
rep_count = 0

last_raw_state = None
raw_streak = 0
stable_state = "unknown"       # 目前穩定狀態
phase = "await_open"           # 'await_open' -> 'saw_open' -> 'saw_open_transit' -> (到 closed 時 +1) -> 'await_open'

cooldown_frames = 2            # 計數後最少等待幀數
since_last_count = cooldown_frames

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

            for i, lm in enumerate(person):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, point_color, -1)
                cv2.putText(frame, f"{i}", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                cv2.putText(frame, f"{lm.x:.2f},{lm.y:.2f}", (cx + 5, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            for start, end in POSE_CONNECTIONS:
                x1, y1 = int(person[start].x * w), int(person[start].y * h)
                x2, y2 = int(person[end].x * w), int(person[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)

            # 分類（原始狀態）
            raw_state, info = classify_jumping_jack(person)

            # 原始狀態連續幀計數，用於「穩定化」
            if raw_state == last_raw_state:
                raw_streak += 1
            else:
                last_raw_state = raw_state
                raw_streak = 1

            # 若原始狀態連續達門檻，更新穩定狀態
            needed = STABLE_FRAMES.get(raw_state, 1)
            if raw_streak >= needed and stable_state != raw_state:
                stable_state = raw_state

                # —— 計數狀態機：必須經過 Open → Transit → Closed 才 +1 ——
                if stable_state == "open":
                    phase = "saw_open"
                elif stable_state == "transit" and phase == "saw_open":
                    phase = "saw_open_transit"
                elif (stable_state == "closed"
                      and phase == "saw_open_transit"
                      and since_last_count >= cooldown_frames):
                    rep_count += 1
                    since_last_count = 0
                    phase = "await_open"
                elif stable_state == "closed":
                    # 沒有經過 open→transit 的 closed，不計數；回到等待 open
                    phase = "await_open"
                else:
                    # 其他組合不影響計數流程
                    pass

            # 冷卻累計
            if since_last_count < cooldown_frames:
                since_last_count += 1

            # 顯示狀態與數值（取鼻點附近顯示）
            px, py = int(person[0].x * w), int(person[0].y * h)
            label = {'open': '展開', 'closed': '合併', 'transit': '過渡'}.get(stable_state, '偵測中')
            color = (0, 255, 0) if stable_state == "open" else ((0, 255, 255) if stable_state == "closed" else (255, 255, 255))
            frame = draw_chinese_text_with_outline(
                frame, f"姿勢（穩定）：{label}", (px, max(py - 70, 0)), font_size=26, text_color=color
            )
            frame = draw_chinese_text_with_outline(
                frame, f"姿勢（原始）：{ {'open':'展開','closed':'合併','transit':'過渡'}.get(raw_state,'偵測中') }",
                (px, max(py - 40, 20)), font_size=20, text_color=(180,180,180)
            )

            # 顯示 腳踝距離 / 髖關節寬度 / 腿部開合比
            la_i, ra_i = 27, 28
            lh_i, rh_i = 23, 24
            xa, ya = int(person[la_i].x * w), int(person[la_i].y * h)
            xb, yb = int(person[ra_i].x * w), int(person[ra_i].y * h)
            xh1, yh1 = int(person[lh_i].x * w), int(person[lh_i].y * h)
            xh2, yh2 = int(person[rh_i].x * w), int(person[rh_i].y * h)

            frame = draw_chinese_text_with_outline(
                frame, f"腳踝距離: {info['ankle_dist']:.2f}",
                (min(xa, xb) - 10, min(ya, yb) - 40), font_size=22
            )
            frame = draw_chinese_text_with_outline(
                frame, f"髖關節寬度: {info['hip_width']:.2f}",
                (min(xh1, xh2) - 10, min(yh1, yh2) - 40), font_size=22
            )
            frame = draw_chinese_text_with_outline(
                frame, f"腿部開合比: {info['ratio']:.2f}",
                (10, 50), font_size=22
            )

            # 左上角顯示總次數
            frame = draw_chinese_text_with_outline(
                frame, f"次數：{rep_count}", (10, 10), font_size=30, text_color=(0, 255, 0)
            )
    else:
        cv2.putText(frame, "No person detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 顯示畫面
    cv2.imshow("Jumping Jacks (Pose + Hand)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ========= 結束處理 =========
cap.release()
cv2.destroyAllWindows()