import cv2
import numpy as np
from skimage.feature import hog
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Flatten
import mediapipe as mp

IMG_SIZE = (128, 128)
CLASSES = ["left", "right", "down", "up"]
MODEL_PATH = "gesture_recognition_model.h5"

def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    features = hog(resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return features


input_shape = 8100
model = Sequential()
model.add(Flatten(input_shape=(input_shape,))) 
model.add(Dense(128, activation='relu')) 
model.add(Dense(64, activation='relu'))  
model.add(Dense(4, activation='softmax')) 

model = keras.models.load_model(MODEL_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
video_path = 'all_gestures_video.mp4'
cap = cv2.VideoCapture(0)


cnt = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_min, x_max, y_min, y_max = frame.shape[1], 0, frame.shape[0], 0
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y

            padding = 20
            hand_roi = frame[y_min-padding:y_max+padding, x_min-padding:x_max+padding]
            predicted_class = ""
            if hand_roi.size != 0:
                hand_roi_resized = cv2.resize(hand_roi, (128, 128))

                hog_features = extract_hog_features(hand_roi_resized)
                hog_features = hog_features.reshape(1, -1)
                prediction = model.predict(hog_features)
                predicted_class_index = np.argmax(prediction)
                predicted_class = CLASSES[predicted_class_index]
                # predicted_class = str(predicted_class_index)
                cv2.putText(frame, predicted_class, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            padding = 20
            cv2.rectangle(frame, (x_min-padding, y_min-padding), (x_max+padding, y_max+padding), (0, 255, 0), 2)
            # if predicted_class != "":
            #     cnt += 1
            #     cv2.imwrite(f"frame_{cnt}.jpg" , frame)
            
    cv2.imshow("Hand Detection", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()