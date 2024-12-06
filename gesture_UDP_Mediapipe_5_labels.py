import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import socket
import time
import os
import sys
import threading

model_path = os.path.join('Assets', 'Scripts', 'Python', 'model_Mediapipe_5_labels', 'model2', 'best_model.h5')

model = tf.keras.models.load_model(model_path)
actions = np.array(['left', 'right', 'down', 'up', 'stop'])

threshold = 0.9

# Thiết lập UDP
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Chuyển ảnh sang RGB và xử lý bằng Mediapipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Vẽ các landmarks của khuôn mặt, cơ thể và tay
def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )

    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )

# Trích xuất các điểm quan trọng từ kết quả Mediapipe
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, face, lh, rh])

# Hiển thị các xác suất cử chỉ với các màu sắc tương ứng
def prob_viz(res, actions, input_image, colors):
    output_image = input_image.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_image, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_image, f'{actions[num]}: {prob:.2f}', (5, 85 + num * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return output_image

# Gửi tín hiệu UDP trong một luồng riêng
def send_udp_signal(gesture_index):
    sock.sendto(int(gesture_index).to_bytes(1, byteorder='big'), (UDP_IP, UDP_PORT))

def main():
    cap = cv2.VideoCapture(0)

    sequence = []
    sentence = []
    last_sent_time = time.time()
    colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 16, 117), (117, 16, 245)]
    waiting_for_stop = False
    gesture_index = 255

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            ret, frame = cap.read()

            image, results = mediapipe_detection(frame, holistic)

            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            # Nếu chưa nhận được cử chỉ "stop", hiển thị None và bỏ qua nhận diện
            if not waiting_for_stop:
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    predicted_label = actions[np.argmax(res)]

                    if predicted_label == 'stop' and res[np.argmax(res)] > threshold:
                        threading.Thread(target=send_udp_signal, args=(4,)).start()  
                        waiting_for_stop = True  
                        sentence = ['stop']
                    else:
                        sentence = []
                else:
                    sentence = []

            # Nếu đã nhận được "stop", chờ cử chỉ tiếp theo để gửi tín hiệu
            elif len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predicted_label = actions[np.argmax(res)]

                if res[np.argmax(res)] > threshold and predicted_label != 'stop':
                    # Gửi tín hiệu và quay lại trạng thái bình thường
                    if predicted_label == 'left':
                        gesture_index = 0
                    elif predicted_label == 'right':
                        gesture_index = 1
                    elif predicted_label == 'down':
                        gesture_index = 2
                    elif predicted_label == 'up':
                        gesture_index = 3
                    elif predicted_label == 'other action':
                        gesture_index = 5

                    # Gửi tín hiệu qua UDP trong một luồng
                    threading.Thread(target=send_udp_signal, args=(gesture_index,)).start()
                    last_sent_time = time.time()
                    waiting_for_stop = False
                    sentence = [predicted_label]

                # Trực quan hóa các nhãn có xác suất vượt ngưỡng
                image = prob_viz([prob if prob > threshold else 0 for prob in res], actions, image, colors)

            # Hiển thị "None" nếu không có cử chỉ nào hoặc đang chờ "stop"
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence) if sentence else 'None',
                        (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Gesture Recognition', image)

            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
