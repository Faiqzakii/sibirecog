import pandas as pd
import numpy as np
import mediapipe as mp

import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from utils import compute_distance, get_mean, get_mean_adj, compute_distance_adj

HANDMARK = mp.solutions.hands.HandLandmark
POSEMARK = mp.solutions.pose.PoseLandmark

# Yang diambil cuma landmark WRIST & TIP [0, 4, 8, 12, 16, 20]
# Selengkapnya: https://google.github.io/mediapipe/images/mobile/hand_landmarks.png
FINGERS = ["THUMB", "INDEX_FINGER", "MIDDLE_FINGER", "RING_FINGER", "PINKY"]
CARTESIAN_FINGERS = [('THUMB', 'INDEX_FINGER'), ('THUMB', 'MIDDLE_FINGER'), 
                     ('THUMB', 'RING_FINGER'), ('THUMB', 'PINKY'), 
                     ('MIDDLE_FINGER', 'INDEX_FINGER'), ('MIDDLE_FINGER', 'RING_FINGER'),
                     ('MIDDLE_FINGER', 'PINKY'), ('PINKY', 'INDEX_FINGER'), 
                     ('PINKY', 'RING_FINGER'), ('RING_FINGER', 'INDEX_FINGER')]

class DataframeLandmark:
    def __init__(self, nb_frames=25):
        self.nb_frames = nb_frames
        self.cols = self.get_col_df_list()
        self.rows = []
        self.tmp_cols = []
        self.data = None
        self.zprevious = {'WRIST':1}
        for finger in FINGERS:
            self.zprevious[finger] = 1

    def __len__(self):
        return len(self.rows)

    def get_col_df_list(self):
        cols = []

        for finger_a, finger_b in CARTESIAN_FINGERS:
            cols.append(f"l_hand_dist_{finger_a}_{finger_b}") #10
        for finger in FINGERS:
            cols.append(f"l_hand_dist_WRIST_{finger}") #5
        for finger_a, finger_b in CARTESIAN_FINGERS:
            cols.append(f"r_hand_dist_{finger_a}_{finger_b}") # 10
        for finger in FINGERS:
            cols.append(f"r_hand_dist_WRIST_{finger}") #5

        # relative distance with origin mean HEAD
        for finger in FINGERS:
            cols.append(f"l_hand_dist_HEAD_{finger}")   #5
        for finger in FINGERS:
            cols.append(f"r_hand_dist_HEAD_{finger}")   #5
        return cols

    def append_landmarks(self, results_hand, results_pose):
        row = []
        tmp_row = []

        # hand landmarks process results_hands.multi_hand_landmarks[0].ListFields()[0][1][20].x
        hands = [hand.label.lower() for hand in results_hand.multi_handedness[0].ListFields()[0][1]]
        pose_points = results_pose.pose_landmarks.ListFields()[0][1]
        landmarks = dict(zip(hands, results_hand.multi_hand_landmarks))

        mean_head = get_mean_adj(
            np.array([pose_points[POSEMARK.RIGHT_EYE_INNER].x, pose_points[POSEMARK.RIGHT_EYE_INNER].y, pose_points[POSEMARK.RIGHT_EYE_INNER].z]),
            np.array([pose_points[POSEMARK.LEFT_EYE_INNER].x, pose_points[POSEMARK.LEFT_EYE_INNER].y, pose_points[POSEMARK.LEFT_EYE_INNER].z]),
            np.array([pose_points[POSEMARK.MOUTH_LEFT].x, pose_points[POSEMARK.MOUTH_LEFT].y, pose_points[POSEMARK.MOUTH_LEFT].z]),
            np.array([pose_points[POSEMARK.MOUTH_RIGHT].x, pose_points[POSEMARK.MOUTH_RIGHT].y, pose_points[POSEMARK.MOUTH_RIGHT].z])
        )

        if landmarks.get("left", False):
            hand_points = landmarks["left"].ListFields()[0][1]
            wrist_point = np.array([hand_points[HANDMARK.WRIST].x, hand_points[HANDMARK.WRIST].y, hand_points[HANDMARK.WRIST].z])
            # COMPUTE DISTANCE BETWEEN FINGER TIPS
            for finger_a, finger_b in CARTESIAN_FINGERS: #10
                #print(hand_points[HANDMARK[f"{finger_a}_TIP"]].z, hand_points[HANDMARK[f"{finger_b}_TIP"]].z)
                point_a = np.array([hand_points[HANDMARK[f"{finger_a}_TIP"]].x, hand_points[HANDMARK[f"{finger_a}_TIP"]].y, hand_points[HANDMARK[f"{finger_a}_TIP"]].z])
                point_b = np.array([hand_points[HANDMARK[f"{finger_b}_TIP"]].x, hand_points[HANDMARK[f"{finger_b}_TIP"]].y, hand_points[HANDMARK[f"{finger_b}_TIP"]].z])
                dist = compute_distance_adj(point_a, point_b)
                row.append(dist)

            # COMPUTE WRIST DISTANCE
            for finger in FINGERS: #5
                finger_tip = np.array([hand_points[HANDMARK[f"{finger}_TIP"]].x, hand_points[HANDMARK[f"{finger}_TIP"]].y, hand_points[HANDMARK[f"{finger}_TIP"]].z])
                row.append(compute_distance_adj(finger_tip, wrist_point))

            # ADD DISTANCE FROM MEAN HEAD
            for finger in FINGERS: #5
                finger_tip = np.array([hand_points[HANDMARK[f"{finger}_TIP"]].x, hand_points[HANDMARK[f"{finger}_TIP"]].y])
                row.append(compute_distance(finger_tip, mean_head))

        else:
            row += np.zeros(10 + 5 + 5).tolist() # 15 frame, 2 dimensi xy, 21 kombinasi landmark tangan + pergelangan

        if landmarks.get("right", False):
            hand_points = landmarks["right"].ListFields()[0][1]
            wrist_point = np.array([hand_points[HANDMARK.WRIST].x, hand_points[HANDMARK.WRIST].y, hand_points[HANDMARK.WRIST].z])
            # COMPUTE DISTANCE BETWEEN FINGER TIPS
            for finger_a, finger_b in CARTESIAN_FINGERS: #10
                #print(hand_points[HANDMARK[f"{finger_a}_TIP"]].z, hand_points[HANDMARK[f"{finger_b}_TIP"]].z)
                point_a = np.array([hand_points[HANDMARK[f"{finger_a}_TIP"]].x, hand_points[HANDMARK[f"{finger_a}_TIP"]].y, hand_points[HANDMARK[f"{finger_a}_TIP"]].z])
                point_b = np.array([hand_points[HANDMARK[f"{finger_b}_TIP"]].x, hand_points[HANDMARK[f"{finger_b}_TIP"]].y, hand_points[HANDMARK[f"{finger_b}_TIP"]].z])
                dist = compute_distance_adj(point_a, point_b)
                row.append(dist)

            # COMPUTE WRIST DISTANCE
            for finger in FINGERS: #5
                finger_tip = np.array([hand_points[HANDMARK[f"{finger}_TIP"]].x, hand_points[HANDMARK[f"{finger}_TIP"]].y, hand_points[HANDMARK[f"{finger}_TIP"]].z])
                row.append(compute_distance_adj(finger_tip, wrist_point))

            # ADD DISTANCE FROM MEAN HEAD
            for finger in FINGERS: #5
                finger_tip = np.array([hand_points[HANDMARK[f"{finger}_TIP"]].x, hand_points[HANDMARK[f"{finger}_TIP"]].y])
                row.append(compute_distance(finger_tip, mean_head))
        else:
            row += np.zeros(10 + 5 + 5).tolist()
        
        self.rows.append(row)
        self.tmp_cols.append(tmp_row)

    def get_dataframe(self):
        if len(self.rows)==0:
            raise Exception('Nothing detected')
        if len(self.rows) < self.nb_frames:
            cpt = 0
            while len(self.rows) < self.nb_frames:
                idx = cpt % (len(self.rows) - 1)
                mean_row = [(value[0] + value[1])/2 for value in zip(self.rows[idx], self.rows[idx + 1])]
                self.rows = self.rows[:idx] + [mean_row] + self.rows[idx:]
                cpt += 2
        elif len(self.rows) > self.nb_frames:
            cpt = 0
            while len(self.rows) > self.nb_frames:
                idx = cpt % (len(self.rows) -1)
                del self.rows[idx + 1]
                cpt += 1
        
        df = pd.DataFrame(self.rows, columns = self.cols)
        self.df = df
        return df
   
    def save_dataframe(self, filepath):
        self.df.to_csv(filepath, index=False)
        
         
class DataRepository:
    def __init__(self, datadir: str):
        self.datadir = datadir
        self.nb_frame = 25
        self.x_train = None
        self.x_val   = None
        self.x_test  = None
        self.y_train = None
        self.y_val   = None
        self.y_test  = None
        self.video_path = None
        self.dataPerWord = []
        self.numClasses = 0
        self.features, self.labels = self.load_data(self.datadir)

    def load_data(self, datadir):
        self.listfile = os.listdir(datadir)
        self.listfile = sorted(self.listfile, key=str.casefold)
        self.numClasses = len(self.listfile)
        # print(self.listfile)
        
        for word in self.listfile:
            if word == '.DS_Store':
                continue
            for csvfile in os.listdir(os.path.join(datadir, word)):
                filepath = os.path.join(datadir, word, csvfile)
                content = pd.read_csv(filepath)
                content = content.reindex(list(range(0, self.nb_frame)), fill_value=0.0)
                content.fillna(0.0, inplace=True)
                self.dataPerWord.append((word, content))
        
        labels   = [n[0] for n in self.dataPerWord]
        features = [n[1] for n in self.dataPerWord]
        features = [f.to_numpy() for f in features]
        
        return features, labels
        
    def getdata(self):
        features = [n[1] for n in self.dataPerWord]
        x = [f.to_numpy() for f in features]
        words = [x.lower() for x in self.listfile]
        
        y = [label.lower() for label in self.labels]
        encoder = LabelBinarizer()
        encoder.fit(words)
        y = encoder.transform(y)
        return np.array(x), np.array(y)
    
    def getDataTrain(self):
        x_train, x_val, y_train, y_val = train_test_split(self.features, self.labels, test_size=0.4, stratify=self.labels) #random_state=42
        x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=42, stratify=y_val)
        words = [x.lower() for x in self.listfile]
        
        y_train = [y.lower() for y in y_train]
        y_test = [y.lower() for y in y_test]
        y_val = [y.lower() for y in y_val]
        
        encoder = LabelBinarizer()
        test = encoder.fit_transform(words)

        y_train = encoder.transform(y_train)
        y_val = encoder.transform(y_val)
        y_test = encoder.transform(y_test)
        
        self.x_train=np.array(x_train)
        self.y_train=np.array(y_train)
        self.x_val=np.array(x_val)
        self.y_val=np.array(y_val)
        self.x_test=np.array(x_test)
        self.y_test=np.array(y_test)
        return self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test, self.labels
    
    def getUnseenX(self, datadir):
        x, _ = self.load_data(datadir)
        return np.array(x)
    
    def countData(self, dataset):
        wordCounter = []
        labels = sorted(set(dataset))
        labels = sorted(labels)
        for label in labels:
            wordCounter.append(0)
        for row in dataset:
            for i in range(len(labels)):
                if str(labels[i]).startswith(row):
                    wordCounter[i] += 1
        for i in range(len(labels)):
            print(labels[i], ': ', wordCounter[i], end =";  ")
            
        print(' ')
    
    def summary(self):
        print('Amount Datasets by word total:', end=' ')
        self.countData(self.labels)
        print('Amount Datasets by word training:', end=' ')
        self.countData(self.y_train)
        print('Amount Datasets by word validiation:', end=' ')
        self.countData(self.y_val)
        print('Amount Datasets by word test:', end=' ')
        self.countData(self.y_test)

        # Display data distribution
        print('\n\nDistribution of data:')
        print("Amount total:", len(self.labels))
        print("Amount training:", len(self.y_train))
        print("Amount validiation:", len(self.y_val))
        print("Amount test:", len(self.y_test))