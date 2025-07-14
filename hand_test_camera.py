import cv2                             # 匯入 OpenCV 模組，用於影像擷取與處理
import mediapipe as mp                 # 匯入 MediaPipe 模組，用於手部偵測
import time                            # 匯入 time 模組，用來取得時間戳（timestamp）
from mediapipe.tasks import python     # 匯入 MediaPipe Tasks Python API 基礎模組
from mediapipe.tasks.python import vision  # 匯入 MediaPipe 的 vision 類別（用於手部模型）

# 模型路徑（這裡使用手部偵測模型檔案）
model_path = "hand_landmarker.task"

# 建立基本模型設定，指定模型檔案路徑
base_options = python.BaseOptions(model_asset_path=model_path)

# 建立手部偵測器的進階設定選項
options = vision.HandLandmarkerOptions(
    base_options=base_options,                  # 使用上面設定的模型路徑
    running_mode=vision.RunningMode.VIDEO,      # 設定為視訊模式（逐幀處理）
    num_hands=2                                  # 每幀最多偵測 2 隻手
)

# 使用設定選項建立一個手部偵測器
detector = vision.HandLandmarker.create_from_options(options)

# 開啟預設攝影機（通常是筆電內建或第一個 USB 攝影機）
cap = cv2.VideoCapture(0)
if not cap.isOpened():                          # 如果攝影機無法開啟就顯示錯誤訊息
    print("❌ 無法開啟攝影機")
    exit()

# 主迴圈：不斷讀取每幀畫面並做偵測
while True:
    ret, frame = cap.read()                     # 從攝影機取得一幀影像
    if not ret:                                 # 如果無法讀取則跳出迴圈
        break

    # OpenCV 是 BGR 預設格式，需轉成 RGB 給 MediaPipe 使用
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 封裝成 MediaPipe 的 Image 格式
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # 計算當前影格的時間戳（毫秒）
    timestamp = int(time.time() * 1000)

    # 對這一幀畫面執行手部偵測
    detection_result = detector.detect_for_video(mp_image, timestamp)

    # 取得影像的尺寸，用來將關鍵點從比例座標轉換為像素座標
    h, w, _ = frame.shape

    # 如果有偵測到手部關鍵點，則逐點畫圈
    if detection_result.hand_landmarks:
        for hand in detection_result.hand_landmarks:     # 每隻手
            for lm in hand:                              # 每個關鍵點（21 點）
                cx, cy = int(lm.x * w), int(lm.y * h)    # 將 0~1 的座標轉為像素
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)  # 在該點畫綠色小圓

    # 顯示加上關鍵點的畫面
    cv2.imshow("Hand Detection", frame)

    # 如果按下鍵盤 q 鍵就結束迴圈
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 釋放攝影機資源與關閉所有視窗
cap.release()
cv2.destroyAllWindows()
