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
VideoSource="精華樹式_BCDvv4UWL8s.mp4" # ← 改為影片來源
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
DEFAULT_FONT_PATH = "C:/Windows/Fonts/msjh.ttc"
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

    # ===== 關鍵：字型快取 =====
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

# ========= Tree Pose 判斷參數（可微調） =========
HIP_LEVEL_EPS = 0.03           # 23與24的y差異，越小越「平」
WRIST_TOUCH_THR = 0.06         # 15與16的距離（相對影像寬度）→「兩手相碰」
ANKLE_HEEL_TOUCH_THR = 0.06    # 點30 與 點27 的距離（相對寬度）→「腳跟碰對側腳踝」
NEAR_THR = 0.08                # 「靠近」判斷的距離（相對寬度）
SMALL_MARGIN = 0.05            # 「下方一點點」的y差（0.05≈5% 影像高）

# 膝角(右腿 24-26-28) 的等級區間（度）
# 樹式越高等 → 右膝彎曲越多（角度越小）
KNEE_ANGLE_L1 = (140, 150)     # Level 1：微彎
KNEE_ANGLE_L2 = (105, 115)     # Level 2：更彎
KNEE_ANGLE_L34 = (35, 45)     # Level 3/4：幾乎成直角到更彎

def norm_dist(a, b, img_w, img_h):
    """歐氏距離，對影像寬做正規化（也可用對角線；這裡用寬較直覺）"""
    dx = (a.x - b.x) * img_w
    dy = (a.y - b.y) * img_h
    return math.hypot(dx, dy) / img_w

def hips_are_level(p23, p24):
    """23、24 y差異小 => 髖水平"""
    return abs(p23.y - p24.y) < HIP_LEVEL_EPS

def wrists_together(p15, p16, img_w, img_h):
    return norm_dist(p15, p16, img_w, img_h) < WRIST_TOUCH_THR

def wrists_up_high(p15, p16, nose, l_sh, r_sh):
    """Level 4 需要「兩手往上舉高舉直」：至少高過肩，最好也高過鼻"""
    cond_shoulder = (p15.y < l_sh.y) and (p16.y < r_sh.y)
    cond_head = (p15.y < nose.y) and (p16.y < nose.y)
    return cond_shoulder and cond_head

def in_angle_range(angle, rng):
    lo, hi = rng
    return (angle >= lo) and (angle <= hi)

# ========= 姿勢分類（含角度偏差） =========
def classify_pose(person, img_w=None, img_h=None):
    """
    樹式四級（自動左右腿）：
    - 共同條件：23-24 水平
    - Level 1：同側膝140°–150°；抬腳跟 ≈ 另一側踝(27 28)相碰
    - Level 2：同側膝105°–115°；抬腳跟在「另一側膝(25 26)」下方一點點
    - Level 3：同側膝35°–45°；抬腳跟靠近「另一側髖(23 24)」
    - Level 4：同 Level 3，但雙手「上舉」（高於肩且高於鼻），不再強制手相碰
    備註：
      - Level 1–3 需要「手相碰」
      - 若左右同時達標，回傳級別較高者；再平手則回傳右腿
    回傳格式：
      ("樹式 Level X (右腿|左腿)", knee_angle) 或 ("非樹式", None)
    """
    try:
        if img_w is None or img_h is None:
            return "非樹式", None

        # 索引
        NOSE = 0
        L_SH, R_SH = 11, 12
        L_HIP, R_HIP = 23, 24
        L_KNEE, R_KNEE = 25, 26
        L_ANK, R_ANK = 27, 28
        L_HEEL, R_HEEL = 29, 30
        L_WR, R_WR = 15, 16

        # 髖水平是必要條件
        if not hips_are_level(person[L_HIP], person[R_HIP]):
            return "非樹式", None

        # 手是否相碰 / 上舉
        hands_together = wrists_together(person[L_WR], person[R_WR], img_w, img_h)
        hands_up = wrists_up_high(person[L_WR], person[R_WR], person[NOSE], person[L_SH], person[R_SH])

        # -------- 右腿抬起（用右膝角，右腳跟對左側骨點）---------
        right_knee_angle = calculate_angle(person[R_HIP], person[R_KNEE], person[R_ANK])
        r_l1 = in_angle_range(right_knee_angle, KNEE_ANGLE_L1) and \
               (norm_dist(person[R_HEEL], person[L_ANK], img_w, img_h) < ANKLE_HEEL_TOUCH_THR) and \
               hands_together
        r_l2 = in_angle_range(right_knee_angle, KNEE_ANGLE_L2) and \
               (person[R_HEEL].y > person[L_KNEE].y) and \
               ((person[R_HEEL].y - person[L_KNEE].y) < SMALL_MARGIN) and \
               hands_together
        r_l3_core = in_angle_range(right_knee_angle, KNEE_ANGLE_L34) and \
                    (norm_dist(person[R_HEEL], person[L_HIP], img_w, img_h) < NEAR_THR)
        r_l3 = r_l3_core and hands_together
        r_l4 = r_l3_core and hands_up

        right_level = 0
        if   r_l4: right_level = 4
        elif r_l3: right_level = 3
        elif r_l2: right_level = 2
        elif r_l1: right_level = 1

        # -------- 左腿抬起（用左膝角，左腳跟對右側骨點）---------
        left_knee_angle = calculate_angle(person[L_HIP], person[L_KNEE], person[L_ANK])
        l_l1 = in_angle_range(left_knee_angle, KNEE_ANGLE_L1) and \
               (norm_dist(person[L_HEEL], person[R_ANK], img_w, img_h) < ANKLE_HEEL_TOUCH_THR) and \
               hands_together
        l_l2 = in_angle_range(left_knee_angle, KNEE_ANGLE_L2) and \
               (person[L_HEEL].y > person[R_KNEE].y) and \
               ((person[L_HEEL].y - person[R_KNEE].y) < SMALL_MARGIN) and \
               hands_together
        l_l3_core = in_angle_range(left_knee_angle, KNEE_ANGLE_L34) and \
                    (norm_dist(person[L_HEEL], person[R_HIP], img_w, img_h) < NEAR_THR)
        l_l3 = l_l3_core and hands_together
        l_l4 = l_l3_core and hands_up

        left_level = 0
        if   l_l4: left_level = 4
        elif l_l3: left_level = 3
        elif l_l2: left_level = 2
        elif l_l1: left_level = 1

        # -------- 選出較高等級；若同等 → 右腿優先 --------
        if right_level == 0 and left_level == 0:
            return "非樹式", None

        if right_level >= left_level:
            return f"樹式 Level {right_level} (右腿)", right_knee_angle
        else:
            return f"樹式 Level {left_level} (左腿)", left_knee_angle

    except:
        return "偵測中...", None


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

            # ---- 先給預設值，避免任何分支/例外造成未定義 ----
            pose_label: str = "非樹式"
            angle_info = None
            display_color = (255, 255, 255)

            # ---- 先做分類（一定要在用到 pose_label 前）----
            try:
                pose_label, angle_info = classify_pose(person, img_w=w, img_h=h)
            except Exception:
                pose_label, angle_info = "偵測中...", None

            # ---- 顏色決定（一定在 pose_label 設定之後）----
            if "Level 4" in pose_label:
                display_color = (0, 200, 255)
            elif "Level 3" in pose_label:
                display_color = (0, 255, 0)
            elif "Level 2" in pose_label:
                display_color = (0, 255, 255)
            elif "Level 1" in pose_label:
                display_color = (0, 165, 255)
            elif "非樹式" in pose_label:
                display_color = (255, 255, 255)
            else:
                display_color = (0, 255, 255)

            # 姿勢標籤（中文）顯示
            frame = draw_chinese_text_with_outline(
                frame, f"姿勢：{pose_label}", (px, max(py - 70, 0)),
                font_path="C:/Windows/Fonts/msjh.ttc",
                font_size=24, text_color=display_color, outline_color=(0,0,0),
                with_outline=True
            )

            if pose_label.startswith("樹式 Level") and angle_info is not None:
                frame = draw_chinese_text_with_outline(
                frame, f"{pose_label}｜膝角約 {angle_info:.0f}°",
                position=(px, max(py - 20, 20)),
                font_path="C:/Windows/Fonts/msjh.ttc",
                font_size=22, text_color=(255,255,255), outline_color=(0,0,0),
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
