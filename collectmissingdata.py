import numpy as np
import os

DATA_PATH = 'MP_Data'
actions = np.array([chr(i) for i in range(65, 91)])  # A–Z
sequence_length = 30
no_sequences = 30

sequences = []
labels = []

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            file_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            if os.path.exists(file_path):
                res = np.load(file_path)
            else:
                print(f"⚠️ Missing file: {file_path} — inserting zeros.")
                res = np.zeros(21 * 3)  # Assuming 21 landmarks, x/y/z = 63 total
            window.append(res)
        sequences.append(window)
        labels.append(np.where(actions == action)[0][0])



import cv2
import os
import numpy as np
import mediapipe as mp

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

DATA_PATH = 'MP_Data'
actions = np.array([chr(i) for i in range(65, 91)])  # A-Z
no_sequences = 30
sequence_length = 30



def extract_keypoints(results):
    if results.multi_hand_landmarks:
        return np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten()
    else:
        return np.zeros(21 * 3)




cap = cv2.VideoCapture(0)

for action in actions:
    for sequence in range(no_sequences):
        for frame_num in range(sequence_length):
            file_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            if os.path.exists(file_path):
                continue  # already exists
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            print(f"Recording missing: {file_path}")
            while True:
                ret, frame = cap.read()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw landmarks
                if results.multi_hand_landmarks:
                    for landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

                # Show frame
                cv2.putText(image, f"Recording {action} Seq:{sequence} Frame:{frame_num}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                cv2.imshow('Re-Recording Missing Frames', image)

                # Wait for 'r' to record this frame
                if cv2.waitKey(1) & 0xFF == ord('r'):
                    keypoints = extract_keypoints(results)
                    np.save(file_path, keypoints)
                    print(f"Saved: {file_path}")
                    break

                # Press 'Esc' to quit
                if cv2.waitKey(1) & 0xFF == 27:
                    print("Esc key pressed. Exiting...")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

cap.release()
cv2.destroyAllWindows()
