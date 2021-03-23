import os

import mediapipe as mp
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

from sign_detector import SignDetector
from utils import get_word_list, get_root_project_path
from dataframe_landmark import DataframeLandmark
from video import VideoStream
from displayer import display_image_landmark


def get_video_list(parent_dir):
    file_paths = []
    for file in os.listdir(parent_dir):
        file_paths.append(os.path.join(parent_dir, file))
    return file_paths


def train_model_from_videos():
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.4, max_num_hands=2)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    model = SignDetector(train=True)
    model.compile()
    dataframes = []
    list_words = get_word_list()
    print("#"*100, "START - training")

    for word_idx in range(0, len(list_words)):
        print("############### START - Train word:", list_words[word_idx])

        for file_path in get_video_list(os.path.join(get_root_project_path(), "data", "video", list_words[word_idx])):
            print("START - video", file_path)
            dfl = DataframeLandmark()
            dfl_flip = DataframeLandmark()
            stream = VideoStream(file_path)
            stream.open()

            for image in stream.get_images():
                results_hands = hands.process(image)
                results_pose = pose.process(image)
                if results_hands.multi_hand_landmarks and results_pose.pose_landmarks:
                    dfl.append_landmarks(results_hands, results_pose)
                    #display_image_landmark(image, results_hands.multi_hand_landmarks, results_pose.pose_landmarks)
                
                # process flip image (nambahin  dataset)
                flip_image = cv2.flip(image, 1)
                results_hands = hands.process(flip_image)
                results_pose = pose.process(flip_image)
                if results_hands.multi_hand_landmarks and results_pose.pose_landmarks:
                    dfl_flip.append_landmarks(results_hands, results_pose)
                    # display_image_landmark(flip_image, results_hands.multi_hand_landmarks, results_pose.pose_landmarks)
            
            stream.close()
            df = dfl.get_dataframe()
            df_flip = dfl_flip.get_dataframe()
            if df is not None:
                df["target"] = word_idx
                dataframes.append(df)
            if df_flip is not None:
                df_flip["target"] = word_idx
                dataframes.append(df_flip)
            
            #print(dataframe)

            print("END - video", file_path)
        print("############### END - Train word:", list_words[word_idx])
    print("#"*100, "END - training")
    
    # merge dataframes
    merged_dataframe = pd.DataFrame([], columns=dataframes[0].columns.values)
    for data in dataframes:
        merged_dataframe = merged_dataframe.append(data)
    targets = merged_dataframe.pop("target")
    model.train(np.array(merged_dataframe), np.array(targets.values.tolist()))


if __name__=="__main__":
    train_model_from_videos()