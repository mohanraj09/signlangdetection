import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
from keras.models import model_from_json
import datetime

# Load model
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

# Actions
actions = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# MediaPipe setup
mp_hands = mp.solutions.handsa
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Log file
log_file = open("predictions_log.txt", "a")

# Extract keypoints (fixed size)
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        keypoints = []
        for landmark in hand.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
    else:
        keypoints = [0] * 63  # Pad with zeros
    return np.array(keypoints)

# Start webcam
cap = cv2.VideoCapture(0)
sequence = []
sentence = []
predictions = []
threshold = 0.8
last_word_spoken = ""

while cap.isOpened():
    ret, frame = cap.read()
    cropframe = frame[40:400, 0:300]
    frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)

    # Detection
    image_rgb = cv2.cvtColor(cropframe, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(cropframe, landmarks, mp_hands.HAND_CONNECTIONS)

    # Extract keypoints and predict
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]

    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        predictions.append(np.argmax(res))
        predictions = predictions[-5:]

        if np.unique(predictions)[0] == np.argmax(res):
            if res[np.argmax(res)] > threshold:
                predicted_action = actions[np.argmax(res)]
                if len(sentence) == 0 or predicted_action != sentence[-1]:
                    sentence.append(predicted_action)

                    # Speak the prediction
                    if predicted_action != last_word_spoken:
                        engine.say(predicted_action)
                        engine.runAndWait()
                        last_word_spoken = predicted_action

                    # Log to file
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f"{timestamp}: {predicted_action}\n")
                    log_file.flush()

        if len(sentence) > 1:
            sentence = sentence[-1:]

    # Display result
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
    cv2.putText(frame, "Output: " + ' '.join(sentence), (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('ASL Detection', frame)

    key = cv2.waitKey(10)
    if key & 0xFF == ord('q') or key == 27:  # 'q' or ESC
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()
