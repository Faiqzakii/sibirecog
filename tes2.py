import os

import mediapipe as mp
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import time
import json
import csv

from sign_detector import SignDetector
from utils import get_word_list
from dataframe_landmark import DataframeLandmark
from streamer import VideoStream
#from displayer import display_image_landmark

datadir = './data/video'

def tes(datadir):
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.4, max_num_hands=2)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    model = SignDetector()
    listwords = get_word_list()
    
    for word in listwords:
        print('{:#^100}'.format(f" START - Train word: {word} "))
        csvpath = os.path.join('./data/csv', word)
        if not os.path.isdir(csvpath):
            os.mkdir(csvpath)
            
        for filename in os.listdir(os.path.join(datadir, word)):
            file_path = os.path.join(datadir, word, filename)
            print("START - video", file_path)
            stream = VideoStream(file_path)
            stream.open()
            
            df = DataframeLandmark()
            dflip = DataframeLandmark()
            
            csvfile = os.path.join(csvpath, f'{filename.replace(".mp4", "")}.csv')
            csvflipfile = os.path.join(csvpath, f'{filename.replace(".mp4", "")}flip.csv')
            # with open(csvfile, 'w', encoding='utf-8') as csvwrite:
            #    writer = csv.writer(csvwrite)
            #    writer.writerow(df.get_col_df_list())
                    
            for image in stream.get_images():
                results_hands = hands.process(image)
                results_pose = pose.process(image)
                if results_hands.multi_hand_landmarks and results_pose.pose_landmarks:
                    df.append_landmarks(results_hands, results_pose)
                
                img_flip = cv2.flip(image, 1) 
                results_hands = hands.process(img_flip)
                results_pose = pose.process(img_flip)
                if results_hands.multi_hand_landmarks and results_pose.pose_landmarks:
                    dflip.append_landmarks(results_hands, results_pose)
                
                    
            stream.close()
            print("END   - video", file_path)
            
            df.get_dataframe()
            df.save_dataframe(csvfile)
            
            dflip.get_dataframe()
            dflip.save_dataframe(csvflipfile)
        print('{:#^100}'.format(f" END - Train word: {word} "))
            
if __name__ == '__main__':
    tes(datadir)
