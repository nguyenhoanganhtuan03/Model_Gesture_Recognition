import cv2
import numpy as np
import tensorflow as tf
import socket
import sys
import io
import mediapipe as mp
import threading
from collections import deque
import time
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

model_path = os.path.join('Assets', 'Scripts', 'Python', 'model_CNN_5_labels', 'Conv3D', 'model_mediapipe', 'best_model.h5')
if not os.path.exists(model_path):
    print(f"File not found: {model_path}")
    
model = tf.keras.models.load_model(model_path)

labels = [
    "down",
    "up",
    "stop",
    "left",
    "right"
]

# Thiết lập UDP
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Bộ nhớ tạm để lưu khung hình
frame_queue = deque(maxlen=2)

stop_threads = False

cap = cv2.VideoCapture(0)

# Biến trạng thái
stop_threads = False
threshold = 0.95
max_prob = 0.0
predicted_label = ""
frames_list = []
waiting_for_stop = False

# Tiền xử lý khung hình: thay đổi kích thước, chuẩn hóa và vẽ các mốc tay
def preprocess_frame(frame, result):
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2))
    frame_resized = cv2.resize(frame, (64, 64))
    frame_normalized = frame_resized / 255.0
    return frame_normalized

# Hàm xử lý khung hình từ webcam
def capture_and_process_frames():
    global stop_threads
    frame_count = 0
    while not stop_threads:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_count % 2 == 0:
            result = hands.process(frame_rgb)
            preprocessed_frame = preprocess_frame(frame, result)
            frame_queue.append((frame, preprocessed_frame))

        if cv2.waitKey(1) & 0xFF == ord('x'):
            stop_threads = True
            break

        frame_count += 1

    cap.release()
    
# Hàm xử lý và gửi dữ liệu cử chỉ qua UDP
def process_and_send_data():
    global stop_threads, predicted_label, frames_list, max_prob, waiting_for_stop
    last_sent_time = time.time()
    gesture_index = 255

    while not stop_threads:
        if frame_queue:
            frame, preprocessed_frame = frame_queue.popleft()
            frames_list.append(preprocessed_frame)
            frames_list = frames_list[-19:] 

            # Kiểm tra nếu đủ 19 frame thì mới tiến hành nhận dạng
            if len(frames_list) == 19:
                res = model.predict(np.expand_dims(frames_list, axis=0))[0]
                predicted_label = labels[np.argmax(res)]
                max_prob = res[np.argmax(res)] 

                if not waiting_for_stop:
                    # Kiểm tra tín hiệu `stop`
                    if predicted_label == 'stop' and max_prob > threshold:
                        sock.sendto(int(4).to_bytes(1, byteorder='big'), (UDP_IP, UDP_PORT))
                        print("Đã gửi tín hiệu stop")
                        waiting_for_stop = True
                        frames_list = [] 
                elif waiting_for_stop:
                    # Sau khi nhận tín hiệu stop, tiếp tục xử lý các tín hiệu khác
                    if max_prob > threshold and predicted_label != 'stop':
                        # Ánh xạ nhãn sang chỉ số
                        if predicted_label == 'left':
                            gesture_index = 0
                        elif predicted_label == 'right':
                            gesture_index = 1
                        elif predicted_label == 'down':
                            gesture_index = 2
                        elif predicted_label == 'up':
                            gesture_index = 3
                        else:  
                            gesture_index = 5

                        # Gửi tín hiệu qua UDP
                        sock.sendto(int(gesture_index).to_bytes(1, byteorder='big'), (UDP_IP, UDP_PORT))
                        print("Đã gửi tín hiệu ", gesture_index)
                        last_sent_time = time.time()
                        waiting_for_stop = False  

            # Hiển thị thông tin cử chỉ và xác suất trên màn hình
            cv2.putText(frame, f"Gesture: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Probability: {max_prob:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            stop_threads = True
            break

    cv2.destroyAllWindows()


# Tạo các luồng
capture_thread = threading.Thread(target=capture_and_process_frames, daemon=True)
process_and_send_thread = threading.Thread(target=process_and_send_data, daemon=True)

capture_thread.start()
process_and_send_thread.start()

capture_thread.join()
process_and_send_thread.join()

cv2.destroyAllWindows()
