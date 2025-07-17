import cv2
import mediapipe as mp
import time
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

# ========== 開啟影片 ==========

#cap = cv2.VideoCapture(0)              # 使用預設攝影機（0 表內建或第一個攝影機）
VideoSource="20250716_Hand+Pose_PostureChecker.MOV" # ← 改為影片來源
cap = cv2.VideoCapture(VideoSource)   
if not cap.isOpened():                # 如果攝影機無法開啟就結束
    print("❌ 無法開啟影片")
    exit()

# 擷取影片資訊
fps = int(cap.get(cv2.CAP_PROP_FPS))
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 準備寫入影片（先預設但不寫入）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = f"{VideoSource.replace('.MOV','')}_OutputAnnotated.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

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

    # 寫入目前這一幀
    out.write(frame)

# ========= 結束處理 =========
cap.release()
cv2.destroyAllWindows()

# 詢問使用者是否要保留儲存影片
save = input("是否要儲存結果影片？(y/n): ").strip().lower()
if save != 'y':
    import os
    os.remove(output_path)
    print("❌ 已刪除影片")
else:
    print("✅ 影片已儲存於：", output_path)

