import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import time
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

video = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while video.isOpened():
        _, frame = video.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)

        image_height, image_width, _ = image.shape
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if results.multi_handedness and results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

                for point in range(len(mp_hands.HandLandmark)):
                    normalized_landmark = hand_landmarks.landmark[point]
                    pixel_coordinates_landmark = mp_drawing._normalized_to_pixel_coordinates(normalized_landmark.x, normalized_landmark.y, image_width, image_height)
                    if mp_hands.HandLandmark(point) == mp_hands.HandLandmark.INDEX_FINGER_TIP:
                        if pixel_coordinates_landmark is not None:
                            cv2.circle(image, (pixel_coordinates_landmark[0], pixel_coordinates_landmark[1]), 25, (0, 200, 0), 5)
                            index_fingertip_x = pixel_coordinates_landmark[0]
                            index_fingertip_y = pixel_coordinates_landmark[1]
                            pyautogui.moveTo(index_fingertip_x * 2, index_fingertip_y * 3)
                            pyautogui.click(button='left')

        cv2.imshow('game', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()




