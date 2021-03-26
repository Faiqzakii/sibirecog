import cv2
import mediapipe as mp
import numpy as np

from sign_detector import SignDetector
from dataframe_landmark import DataframeLandmark


def display_from_stream(stream, mp_pose, mp_hands):
    stream.open()
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.4, max_num_hands=2)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.4)
    for img in stream.get_images():
        try:
            results_hands = hands.process(img)
            results_pose = pose.process(img)
            img.flags.writeable = True
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    img, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow('MediaPipe Hands', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as err:
            print(err)
            break
    stream.close()
    hands.close()
    pose.close()


def display_evaluate_from_stream(stream, mp_pose, mp_hands, model):
    model = SignDetector(filepath=model)
    dfl = DataframeLandmark()
    stream.open()
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.4, max_num_hands=2)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # predicted_word = "None"

    for img in stream.get_images():
        results_hands = hands.process(img)
        results_pose = pose.process(img)
        img.flags.writeable = True
        if results_hands.multi_hand_landmarks and results_pose.pose_landmarks:
            dfl.append_landmarks(results_hands, results_pose)
    
    dataframe = dfl.get_dataframe()
    predicted_word, prob = model.evaluate(np.array(dataframe))
    print('#'*50)
    print('{:#^50}'.format(f" Prediction:{predicted_word} "))
    print('#'*50)

    stream.open()
    for img in stream.get_images():
        results_hands = hands.process(img)
        results_pose = pose.process(img)
        if results_hands.multi_hand_landmarks and results_pose.pose_landmarks:
            display_image_landmark(img, results_hands.multi_hand_landmarks, results_pose.pose_landmarks, word=predicted_word, prob=prob, width=stream.getwidth())
    stream.close()
    hands.close()
    pose.close()


def display_image_landmark(image, hand_multi_landmarks, pose_landmarks, word=None, prob=None, width=0):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    image.flags.writeable = True
    for hand_landmark in hand_multi_landmarks:
        mp_drawing.draw_landmarks(
            image, hand_landmark, mp_hands.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if word is not None:
        cv2.putText(image, f'{word} : {prob:.3f}', (int(width/5), 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pass
