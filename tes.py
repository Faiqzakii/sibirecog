import cv2
import mediapipe as mp
import numpy as np

from utils import compute_distance, get_mean

def run_simple():
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    handsd = mp_hands.Hands(
        min_detection_confidence=0.7, min_tracking_confidence=0.5)
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture('./test/kapan_t.mp4')
    while cap.isOpened():
        success, image = cap.read()
        x = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        y = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        image = cv2.resize(image, (int(x/3),int(y/3)))
        if not success:
            break
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results_hands = handsd.process(image)
        results_pose = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results_hands.multi_hand_landmarks and results_pose.pose_landmarks:
            # for hand_landmark in results.multi_hand_landmarks:
            #    print(hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP])
            
            # print(results_hands.multi_hand_landmarks)
            # print([a for a in dir(results_hands.multi_hand_landmarks) if not a.startswith('_')])
            # print([a for a in dir(results_pose.pose_landmarks) if not a.startswith('_')])
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # append_landmarks(results_hands, results_pose)
        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
    
    handsd.close()
    pose.close()
    cap.release()


run_simple()
