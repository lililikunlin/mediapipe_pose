import cv2                             # OpenCV：用於攝影機與影像顯示處理
import mediapipe as mp                 # MediaPipe：Google 提供的姿勢估計模型
import time                            # 用於取得當前時間（timestamp）
from mediapipe.tasks import python     # 匯入 MediaPipe 的 Python Tasks API
from mediapipe.tasks.python import vision  # 匯入 MediaPipe 的視覺任務模組

# ========== 模型初始化 ==========

model_path = 'pose_landmarker_full.task'  # 指定模型檔案路徑

# 設定模型的基本選項（如模型路徑）
base_options = python.BaseOptions(model_asset_path=model_path)

# 建立 PoseLandmarker 的設定參數
options = vision.PoseLandmarkerOptions(
    base_options=base_options,                     # 使用指定的模型
    running_mode=vision.RunningMode.VIDEO,         # 運行模式：視訊（逐幀處理）
    output_segmentation_masks=False,               # 是否輸出遮罩（這裡關閉）
    num_poses=3,                                   # 最多同時偵測 3 個人
    min_pose_detection_confidence=0.6,             # 姿勢初步偵測最小信心值
    min_pose_presence_confidence=0.6,              # 姿勢存在判定最小信心值
    min_tracking_confidence=0.5                    # 追蹤階段的最小信心值
)

# 根據參數建立姿勢偵測器
detector = vision.PoseLandmarker.create_from_options(options)

# ========== 骨架點對點連線表（根據 Mediapipe Pose 規則） ==========

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),  # 眼睛
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # 手臂
    (15, 17), (16, 18),
    (11, 23), (12, 24), (23, 24),  # 軀幹
    (23, 25), (25, 27), (27, 29), (29, 31),  # 腿
    (24, 26), (26, 28), (28, 30), (30, 32)
]

# 定義每個人的顏色組合（點顏色, 線顏色）
PERSON_COLORS = [
    ((0, 255, 0), (0, 255, 255)),     # 綠點 + 黃骨架
    ((255, 0, 0), (255, 255, 0)),     # 藍點 + 青骨架
    ((0, 0, 255), (255, 0, 255))      # 紅點 + 紫骨架
]

# ========== 開啟影片 ==========

#cap = cv2.VideoCapture(0)              # 使用預設攝影機（0 表內建或第一個攝影機）
cap = cv2.VideoCapture("20250714.MOV")   # ← 改為影片來源
if not cap.isOpened():                # 如果攝影機無法開啟就結束
    print("❌ 無法開啟影片")
    exit()

# 擷取影片資訊
fps = int(cap.get(cv2.CAP_PROP_FPS))
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 準備寫入影片（先預設但不寫入）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = "output_annotated.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# ========== 主迴圈開始：逐幀讀取畫面與姿勢分析 ==========

while True:
    ret, frame = cap.read()           # 讀取一幀畫面
    if not ret:
        break                         # 若無畫面就跳出

    # OpenCV 為 BGR，MediaPipe 需要 RGB
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 將影像轉換為 MediaPipe 的格式
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # 計算當前幀的時間戳（毫秒）
    timestamp = int(time.time() * 1000)

    # 執行偵測：針對目前這幀圖像做姿勢預測
    detection_result = detector.detect_for_video(mp_image, timestamp)

    # 取得影像的高寬（用於將 0~1 座標轉為實際像素位置）
    h, w, _ = frame.shape

    # ========== 當沒偵測到任何姿勢時，顯示警告 ==========

    if not detection_result.pose_landmarks:
        cv2.putText(frame, "No person detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ========== 若偵測到多個人，則處理每個人 ==========
    else:
        for idx, person_landmarks in enumerate(detection_result.pose_landmarks):
            # 根據 index 指派點與線的顏色（輪流使用）
            point_color, line_color = PERSON_COLORS[idx % len(PERSON_COLORS)]

            # 取得頭部（landmark 0）位置，用來標記「Person n」
            first_lm = person_landmarks[0]
            px, py = int(first_lm.x * w), int(first_lm.y * h)
            cv2.putText(frame, f"Person {idx + 1}", (px, py-80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, point_color, 2)

            # 畫出每個關鍵點與其資訊（index 與座標）
            for i, lm in enumerate(person_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, point_color, -1)  # 畫點
                cv2.putText(frame, f"{i}", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                cv2.putText(frame, f"{lm.x:.2f},{lm.y:.2f}", (cx + 5, cy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            # 根據 POSE_CONNECTIONS 畫出骨架連線
            for start_idx, end_idx in POSE_CONNECTIONS:
                x1, y1 = int(person_landmarks[start_idx].x * w), int(person_landmarks[start_idx].y * h)
                x2, y2 = int(person_landmarks[end_idx].x * w), int(person_landmarks[end_idx].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)

    # 顯示結果視窗
    cv2.imshow("Pose Detection (Multi-person)", frame)
    # ✅ 寫入目前這一幀
    out.write(frame)

    # 如果按下 'q' 鍵就跳出迴圈
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ========== 清理資源 ==========
cap.release()              # 關閉攝影機
cv2.destroyAllWindows()    # 關閉 OpenCV 視窗

# 詢問使用者是否要保留儲存影片
save = input("是否要儲存結果影片？(y/n): ").strip().lower()
if save != 'y':
    import os
    os.remove(output_path)
    print("❌ 已刪除影片")
else:
    print("✅ 影片已儲存於：", output_path)
