import cv2
import mediapipe as mp
import numpy as np

from utils import compute_distance, get_mean

HANDMARK = mp.solutions.hands.HandLandmark
POSEMARK = mp.solutions.pose.PoseLandmark

FINGERS = ["THUMB", "INDEX_FINGER", "MIDDLE_FINGER", "RING_FINGER", "PINKY"]
CARTESIAN_FINGERS = [('THUMB', 'INDEX_FINGER'), ('THUMB', 'MIDDLE_FINGER'), ('THUMB', 'RING_FINGER'), ('THUMB', 'PINKY'),
                     ('MIDDLE_FINGER', 'INDEX_FINGER'), ('MIDDLE_FINGER', 'RING_FINGER'),
                     ('MIDDLE_FINGER', 'PINKY'), ('PINKY', 'INDEX_FINGER'), ('PINKY', 'RING_FINGER'),
                     ('RING_FINGER', 'INDEX_FINGER')]

def append_landmarks(results_hand, results_pose):
    row = []
    tmp_row = []

    # hand landmarks process results_hands.multi_hand_landmarks[0].ListFields()[0][1][20].x
    hands = [hand.label.lower() for hand in results_hand.multi_handedness[0].ListFields()[0][1]]
    pose_points = results_pose.pose_landmarks.ListFields()[0][1]
    landmarks = dict(zip(hands, results_hand.multi_hand_landmarks))

    mean_head = get_mean(
        np.array([pose_points[POSEMARK.RIGHT_EYE_INNER].x, pose_points[POSEMARK.RIGHT_EYE_INNER].y, pose_points[POSEMARK.RIGHT_EYE_INNER].z]),
        np.array([pose_points[POSEMARK.LEFT_EYE_INNER].x, pose_points[POSEMARK.LEFT_EYE_INNER].y, pose_points[POSEMARK.LEFT_EYE_INNER].z]),
        np.array([pose_points[POSEMARK.MOUTH_LEFT].x, pose_points[POSEMARK.MOUTH_LEFT].y,
                pose_points[POSEMARK.MOUTH_LEFT].z]),
        np.array([pose_points[POSEMARK.MOUTH_RIGHT].x, pose_points[POSEMARK.MOUTH_RIGHT].y,
                pose_points[POSEMARK.MOUTH_RIGHT].z])
    )

    if landmarks.get("left", False):
        hand_points = landmarks["left"].ListFields()[0][1]
        wrist_point = np.array([hand_points[HANDMARK.WRIST].x, hand_points[HANDMARK.WRIST].y, hand_points[HANDMARK.WRIST].z])
        # COMPUTE DISTANCE BETWEEN FINGER TIPS (10)
        for finger_a, finger_b in CARTESIAN_FINGERS:
            point_a = np.array([hand_points[HANDMARK[f"{finger_a}_TIP"]].x, hand_points[HANDMARK[f"{finger_a}_TIP"]].y,
                                hand_points[HANDMARK[f"{finger_a}_TIP"]].z])
            point_b = np.array([hand_points[HANDMARK[f"{finger_b}_TIP"]].x, hand_points[HANDMARK[f"{finger_b}_TIP"]].y,
                                hand_points[HANDMARK[f"{finger_b}_TIP"]].z])
            dist = compute_distance(point_a, point_b)
            row.append(dist)


        # COMPUTE WRIST DISTANCE (5)
        for finger in FINGERS:
            finger_tip = np.array([hand_points[HANDMARK[f"{finger}_TIP"]].x, hand_points[HANDMARK[f"{finger}_TIP"]].y,
                                hand_points[HANDMARK[f"{finger}_TIP"]].z])
            row.append(compute_distance(finger_tip, wrist_point))
        
        # ADD RELATIVE COORDINATE FROM MEAN HEAD (63)
        for landmark in landmarks["left"].ListFields()[0][1]:
            row += [landmark.x - mean_head[0], landmark.y - mean_head[1], landmark.z - mean_head[2]]

    else:
        row += np.zeros(15 + 3 * 21).tolist() # 15 frame, 3 dimensi xyz, 21 kombinasi landmark tangan

    if landmarks.get("right", False):
        hand_points = landmarks["right"].ListFields()[0][1]
        wrist_point = np.array([hand_points[HANDMARK.WRIST].x, hand_points[HANDMARK.WRIST].y, hand_points[HANDMARK.WRIST].z])
        # COMPUTE DISTANCE BETWEEN FINGER TIPS (10)
        for finger_a, finger_b in CARTESIAN_FINGERS:
            point_a = np.array([hand_points[HANDMARK[f"{finger_a}_TIP"]].x, hand_points[HANDMARK[f"{finger_a}_TIP"]].y,
                                hand_points[HANDMARK[f"{finger_a}_TIP"]].z])
            point_b = np.array([hand_points[HANDMARK[f"{finger_b}_TIP"]].x, hand_points[HANDMARK[f"{finger_b}_TIP"]].y,
                                hand_points[HANDMARK[f"{finger_b}_TIP"]].z])
            dist = compute_distance(point_a, point_b)
            row.append(dist)
        
        # COMPUTE WRIST DISTANCE (5)
        for finger in FINGERS:
            finger_tip = np.array(
                [hand_points[HANDMARK[f"{finger}_TIP"]].x, hand_points[HANDMARK[f"{finger}_TIP"]].y,
                hand_points[HANDMARK[f"{finger}_TIP"]].z])
            row.append(compute_distance(finger_tip, wrist_point))

        # ADD RELATIVE COORDINATE FROM MEAN HEAD (63)
        for landmark in landmarks["right"].ListFields()[0][1]:
            row += [landmark.x - mean_head[0], landmark.y - mean_head[1], landmark.z - mean_head[2]]
    else:
        row += np.zeros(15 + 3 * 21).tolist()

    # pose landmarks process (33*3xyz)
    for landmark in pose_points:
        row += [landmark.x, landmark.y, landmark.z]

    #print(row)
    #print(len(row))

def run_simple():
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    handsd = mp_hands.Hands(
        min_detection_confidence=0.7, min_tracking_confidence=0.5)
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
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
            
            #print(results.multi_hand_landmarks)
            
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            append_landmarks(results_hands, results_pose)
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