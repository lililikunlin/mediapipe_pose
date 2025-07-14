import cv2                             # 匯入 OpenCV 模組，處理影像擷取與顯示
import mediapipe as mp                 # 匯入 MediaPipe 模組，進行手部關鍵點偵測
import time                            # 匯入 time 模組，用於取得時間戳
from mediapipe.tasks import python     # 匯入 MediaPipe Tasks API 的 Python 基礎模組
from mediapipe.tasks.python import vision  # 匯入 MediaPipe 的視覺任務模組（如手部偵測）

# ========== 模型初始化 ==========

model_path = "hand_landmarker.task"   # 指定手部偵測模型檔案路徑

# 設定模型的基本參數（如模型路徑）
base_options = python.BaseOptions(model_asset_path=model_path)

# 手部偵測器的進階設定：偵測模式與最多幾隻手
options = vision.HandLandmarkerOptions(
    base_options=base_options,                  # 套用基本參數
    running_mode=vision.RunningMode.VIDEO,      # 設定為視訊模式（逐幀處理）
    num_hands=2                                 # 每幀最多偵測 2 隻手
)

# 建立手部偵測器
detector = vision.HandLandmarker.create_from_options(options)

# ========== 攝影機初始化 ==========

cap = cv2.VideoCapture(0)             # 開啟預設攝影機（0 表內建或第一個攝影機）
if not cap.isOpened():                # 如果無法開啟攝影機，顯示錯誤並結束程式
    print("❌ 無法開啟攝影機")
    exit()

# ========== 定義連線與顏色 ==========

# 定義手部 21 點的骨架連線方式（參考 MediaPipe 官方定義）
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),            # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),       # 中指
    (9, 13), (13, 14), (14, 15), (15, 16),     # 無名指
    (13, 17), (17, 18), (18, 19), (19, 20)     # 小指
]

# 每隻手使用不同的顏色（點顏色, 線顏色）
HAND_COLORS = [
    ((0, 255, 0), (0, 255, 255)),     # 第 1 隻手：綠點 + 黃線
    ((255, 0, 0), (255, 255, 0)),     # 第 2 隻手：藍點 + 青線
    ((0, 0, 255), (255, 0, 255))      # 預備第 3 隻手（若啟用）
]

# ========== 主迴圈：持續讀取每一幀影像 ==========

while True:
    ret, frame = cap.read()              # 從攝影機讀取一幀影像
    if not ret:
        break                            # 若讀取失敗則結束迴圈

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # OpenCV 預設 BGR，要轉為 RGB 給 MediaPipe 用
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)  # 封裝為 MediaPipe 格式

    timestamp = int(time.time() * 1000)  # 取得時間戳（單位：毫秒）
    detection_result = detector.detect_for_video(mp_image, timestamp)  # 偵測當前畫面

    h, w, _ = frame.shape  # 取得影像的高與寬，用來換算座標比例

    # 如果有偵測到手
    if detection_result.hand_landmarks:
        for idx, hand in enumerate(detection_result.hand_landmarks):  # 對每一隻手進行處理
            point_color, line_color = HAND_COLORS[idx % len(HAND_COLORS)]  # 根據第幾隻手選擇顏色

            # 畫出每一個關鍵點（21 點）
            for i, lm in enumerate(hand):
                cx, cy = int(lm.x * w), int(lm.y * h)  # 轉換為畫面像素座標
                cv2.circle(frame, (cx, cy), 4, point_color, -1)  # 畫點
                cv2.putText(frame, f"{i}", (cx + 5, cy - 5),     # 顯示點的編號
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                cv2.putText(frame, f"{lm.x:.2f},{lm.y:.2f}",     # 顯示點的座標
                            (cx + 5, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            # 畫出手指之間的骨架連線
            for start, end in HAND_CONNECTIONS:
                x1, y1 = int(hand[start].x * w), int(hand[start].y * h)
                x2, y2 = int(hand[end].x * w), int(hand[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)

    # 顯示影像視窗
    cv2.imshow("Hand Detection with Landmarks & Connections", frame)

    # 按下 q 鍵即可退出迴圈
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ========== 資源釋放 ==========

cap.release()               # 關閉攝影機
cv2.destroyAllWindows()     # 關閉所有 OpenCV 視窗
