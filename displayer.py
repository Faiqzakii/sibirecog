import cv2
import mediapipe as mp
import numpy as np
import time
import csv

from sign_detector import SignDetector
from dataframe_landmark import DataframeLandmark

def display_from_stream(vstream, mp_pose, mp_hands):
    vstream.open()
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.4, max_num_hands=2)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.4)
    for img in vstream.get_images():
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
            cv2.imshow('BISINDO Translator', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as err:
            print(err)
            break
    vstream.close()
    hands.close()
    pose.close()


def display_evaluate_from_stream(stream, mp_pose, mp_hands, mpath):
    start = time.time()
    model = SignDetector(filepath=mpath)
    dfl = DataframeLandmark()
    stream.open()
    vpath = stream.get_path()
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.4, max_num_hands=2)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    for img in stream.get_images():
        results_hands = hands.process(img)
        results_pose = pose.process(img)
        img.flags.writeable = True
        if results_hands.multi_hand_landmarks and results_pose.pose_landmarks:
            dfl.append_landmarks(results_hands, results_pose)

    dataframe = dfl.get_dataframe()
    dataframe = dataframe.to_numpy()
    dataframe = dataframe.reshape(1, dataframe.shape[0], dataframe.shape[1])
    predicted_word, prob = model.evaluate(dataframe)
    print('#'*50)
    print('{:#^50}'.format(f" Prediction:{predicted_word} "))
    print('{:#^50}'.format(f" Confidence:{prob:.3f} "))
    print('#'*50)
    
    with open('./out/runtime.csv', mode='a', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([vpath, mpath, predicted_word, prob, time.time()-start])
         
    stream.open()
    cnt = 0
    for img in stream.get_images():
        results_hands = hands.process(img)
        results_pose = pose.process(img)
        if results_hands.multi_hand_landmarks and results_pose.pose_landmarks:
            display_image_landmark(img, cnt, results_hands.multi_hand_landmarks, results_pose.pose_landmarks, word=predicted_word, prob=prob, width=stream.getwidth())
            cnt += 1
    stream.close()
    hands.close()
    pose.close()
    

def display_image_landmark(image, cnt, hand_multi_landmarks, pose_landmarks, word=None, prob=None, width=0):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        width = width/3
        cv2.putText(image, f'{word} : {prob:.3f}', (int(width/5), 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('BISINDO Translator', image)
    cv2.imwrite("./out/frame%d.jpg" % cnt, image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pass
