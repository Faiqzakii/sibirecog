import cv2
import mediapipe as mp
import numpy as np
import os
import csv

from utils import compute_distance, get_word_list
from dataframe_landmark import DataframeLandmark
from streamer import VideoStream

HANDMARK = mp.solutions.hands.HandLandmark

def run_simple():
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    handsd = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture('./test/peraga2.mp4')
    cnt = 0
    while cap.isOpened():
        success, image = cap.read()
        x = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        y = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if not success:
            break
            
        image = cv2.resize(image, (int(x/3),int(y/3)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results_hands = handsd.process(image)
        results_pose = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results_hands.multi_hand_landmarks and results_pose.pose_landmarks:
            hands = [hand.label.lower() for hand in results_hands.multi_handedness[0].ListFields()[0][1]]
            landmarks = dict(zip(hands, results_hands.multi_hand_landmarks))
            dist = 0
            
            if landmarks.get("left", False):
                hand_point = landmarks['left'].ListFields()[0][1]
                thumb = np.array([hand_point[HANDMARK["THUMB_TIP"]].x, hand_point[HANDMARK['THUMB_TIP']].y])
                indfing = np.array([hand_point[HANDMARK["INDEX_FINGER_TIP"]].x, hand_point[HANDMARK['INDEX_FINGER_TIP']].y])
                dist = compute_distance(thumb, indfing)
                print(dist)

            if landmarks.get("right", False):
                hand_point = landmarks['right'].ListFields()[0][1]
                thumb = np.array([hand_point[HANDMARK["THUMB_TIP"]].x, hand_point[HANDMARK['THUMB_TIP']].y])
                indfing = np.array([hand_point[HANDMARK["INDEX_FINGER_TIP"]].x, hand_point[HANDMARK['INDEX_FINGER_TIP']].y])
                dist = compute_distance(thumb, indfing)
                print(dist)
                
            """ for hand_landmark in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmark, mp_hands.HAND_CONNECTIONS)
                cv2.putText(image, f'Distance Thumb-Index : {dist:.4f}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
             """    
            cv2.imwrite("./out/polosan/polos/frame%d.jpg" % cnt, image)
            #print("./out/polosan/skeleton/frame%d.jpg" % cnt)
            
            # print([a for a in dir(results_hands.multi_hand_landmarks) if not a.startswith('_')])
            # print([a for a in dir(results_pose.pose_landmarks) if not a.startswith('_')])
                

        #if results_pose.pose_landmarks:
        #    mp_drawing.draw_landmarks(image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        

        cv2.imshow('MediaPipe Hands', image)

        cnt += 1
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
    # print(landmarks['right'].ListFields()[0][1][HANDMARK["THUMB_TIP"]])
    handsd.close()
    pose.close()
    cap.release()

def run_holistic():
    mp_holi = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    holi = mp_holi.Holistic(min_detection_confidence=0.5)
    cap = cv2.VideoCapture('./test/berapat2.mp4')
    while cap.isOpened():
        success, image = cap.read()
        x = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        y = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if not success:
            break
            
        image = cv2.resize(image, (int(x/3),int(y/3)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        result_holi = holi.process(image)
        
        print([a for a in dir(result_holi.left_hand_landmarks) if not a.startswith('_')])
    holi.close()
    cap.release()

def frame_videos(datadir = './data/video'):
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.4, max_num_hands=2)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    listwords = get_word_list()
    csvdir = './out'
    
    csvpath = os.path.join(csvdir, 'frame.csv')
    with open(csvpath, 'w', encoding='utf-8', newline='') as csvwrite:
            writer = csv.writer(csvwrite)
            writer.writerow(['filename', 'frame']) 
            
    for word in listwords:
        print('{:#^100}'.format(f" START - Read word: {word} "))
            
        for filename in os.listdir(os.path.join(datadir, word)):
            file_path = os.path.join(datadir, word, filename)
            
            stream = VideoStream(file_path)
            stream.open()
            
            df = DataframeLandmark()
            dflip = DataframeLandmark()
            
            for img in stream.get_images():
                results_hands = hands.process(img)
                results_pose = pose.process(img)
                if results_hands.multi_hand_landmarks and results_pose.pose_landmarks:
                    df.append_landmarks(results_hands, results_pose)
                    
                img_flip = cv2.flip(img, 1)
                results_hands = hands.process(img_flip)
                results_pose = pose.process(img_flip)
                if results_hands.multi_hand_landmarks and results_pose.pose_landmarks:
                    dflip.append_landmarks(results_hands, results_pose)
                    
            stream.close()
            
            with open(csvpath, 'a', encoding='utf-8', newline = '') as csvwrite:
               writer = csv.writer(csvwrite)
               writer.writerow([filename, len(df.rows)])
               writer.writerow([filename + " - flip", len(dflip.rows)])
            
            print(f'{filename}       : {len(df.rows)}')
            print(f'{filename + " - flip"}: {len(dflip.rows)}')
        print('{:#^100}'.format(f" END - Read word: {word} "))
      
run_simple()
