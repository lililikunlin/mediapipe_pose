# 匯入所需套件
import cv2                             # OpenCV：處理影像與視訊
import mediapipe as mp                 # MediaPipe：人體姿勢偵測等
import time                            # 用來取得時間戳（timestamp）
from mediapipe.tasks import python     # 匯入 MediaPipe Tasks API
from mediapipe.tasks.python import vision  # 匯入視覺類別（PoseLandmarker 在這裡）

# ========== 模型初始化 ==========

model_path = 'pose_landmarker_full.task'  # 指定姿勢偵測模型檔案路徑

# 設定基本選項，包括模型路徑
base_options = python.BaseOptions(model_asset_path=model_path)

# 建立 PoseLandmarker 選項
options = vision.PoseLandmarkerOptions(
    base_options=base_options,                     # 使用的模型設定
    running_mode=vision.RunningMode.VIDEO,         # 選擇 VIDEO 模式（適合逐幀影像）
    output_segmentation_masks=False,               # 不輸出人體區域遮罩
    num_poses=3,                                   # 最多可偵測幾個人（此處設定為 3 人）
    min_pose_detection_confidence=0.6,             # 最小姿勢偵測信心值
    min_pose_presence_confidence=0.6,              # 最小姿勢存在信心值
    min_tracking_confidence=0.5                    # 最小追蹤信心值
)

# 建立 PoseLandmarker 偵測器
detector = vision.PoseLandmarker.create_from_options(options)

# ========== 骨架線段對應關係 ==========

POSE_CONNECTIONS = [
    # 頭部與手臂連線
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),  # 眼睛之間
    # 上半身與手部
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (16, 18),
    # 軀幹與腿部
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32)
]

# ========== 開啟攝影機 ==========

cap = cv2.VideoCapture(0)              # 開啟攝影機（設備編號 0）
if not cap.isOpened():                # 如果打不開，則報錯並退出
    print("❌ 無法開啟攝影機")
    exit()

# ========== 主迴圈：持續讀取每一幀影像 ==========

while True:
    ret, frame = cap.read()           # 讀取一幀影像
    if not ret:
        break                         # 如果沒讀到畫面就結束

    # 轉成 RGB 格式（MediaPipe 需要 RGB）
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 將 OpenCV 的影像封裝成 MediaPipe 的格式
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # 使用 detect_for_video 模式（需提供時間戳）
    timestamp = int(time.time() * 1000)                   # 取得時間戳（單位毫秒）
    detection_result = detector.detect_for_video(mp_image, timestamp)

    # 取得影像的寬與高（用來計算座標轉換）
    h, w, _ = frame.shape

    # ========= 如果偵測到姿勢，畫出骨架與點 =========
    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]    # 只取第 1 個人（index=0）

        # 畫每個關鍵點（33 個點）
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)         # 將 0~1 座標轉換為畫面上的 pixel
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)  # 畫綠色實心圓

        # 畫骨架連線（根據上面定義好的 index 對應表）
        for start_idx, end_idx in POSE_CONNECTIONS:
            x1, y1 = int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h)
            x2, y2 = int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # 黃綠線
    else:
        # 沒有人被偵測到時，顯示紅字提示
        cv2.putText(frame, "No person detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ========= 顯示每個關鍵點的編號與座標 =========
    for person_landmarks in detection_result.pose_landmarks:     # 支援多人體
        for i, landmark in enumerate(person_landmarks):          # 每一個點（共 33 個）
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)       # 再畫一次點（可省略）
            cv2.putText(frame, f"{i}", (cx + 5, cy - 5),          # 顯示編號
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame, f"{landmark.x:.2f},{landmark.y:.2f}",  # 顯示 (x, y) 座標
                        (cx + 5, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # 顯示視窗
    cv2.imshow("Pose Detection (with Skeleton)", frame)

    # 按 q 鍵中斷並結束
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ========== 清除資源 ==========
cap.release()                    # 關閉攝影機
cv2.destroyAllWindows()          # 關閉所有 OpenCV 視窗
