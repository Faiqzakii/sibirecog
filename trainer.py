import os

import mediapipe as mp
import cv2

from sign_detector import SignDetector
from utils import get_word_list, plot
from dataframe_landmark import DataframeLandmark, DataRepository
from streamer import VideoStream
import pandas as pd
import time

"""
Sengaja di bikin doc sebagai referensi V1

def train_model_from_videos(modelname='tes', epoch=250):
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.4, max_num_hands=2)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    model = SignDetector()

    dataframes = []
    list_words = get_word_list()
    print('{:#^100}'.format(" START - training "))

    for word_idx in range(0, len(list_words)):
        print('{:#^100}'.format(f" START - Train word: {list_words[word_idx]} "))

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

            print("END   - video", file_path)
        print('{:#^100}'.format(f" END - Train word: {list_words[word_idx]} "))
    print('{:#^100}'.format(" END - training "))
    
    # merge dataframes
    merged_dataframe = pd.DataFrame([], columns=dataframes[0].columns.values)
    for data in dataframes:
        merged_dataframe = merged_dataframe.append(data)
    merged_dataframe.to_csv('./data/tes.csv')
    # print(np.array(merged_dataframe).shape)
    targets = merged_dataframe.pop("target")
    history = model.train(np.array(merged_dataframe), np.array(targets.values.tolist()), epochs=epoch)
    plot(history)
    json.dump(history.history, open(f'./data/{modelname}.json', 'w'))
    
    model.save(modelname=modelname)
    print(f'Training model saved in ./model/{modelname}.h5')
"""

def videos_tocsv(datadir = './data/video'):
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.4, max_num_hands=2)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    listwords = get_word_list()
    csvdir = './data/csv'
    
    for word in listwords:
        print('{:#^100}'.format(f" START - Read word: {word} "))
        csvpath = os.path.join(csvdir, word)
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
            print("END   - video", file_path)
            
            df.get_dataframe()
            df.save_dataframe(csvfile)
            dflip.get_dataframe()
            dflip.save_dataframe(csvflipfile)
        print('{:#^100}'.format(f" END - Read word: {word} "))
    
    print('{:#^100}'.format(f" All video successfully converted into csv: {csvdir} "))          
            
def train_from_videos(datadir = './data/csv', modelname = 'tes', epochs = 50):
    videos_tocsv()
    
    repo = DataRepository(datadir=datadir)
    ny = repo.numClasses
    
    model = SignDetector()
    X_train, X_val, X_test, y_train, y_val, y_test, labels = repo.getDataTrain()
    
    arch = 'seqlstm3'
    model.create_model(X_train, ny, arch)
    history = model.train(X_train, y_train, X_val, y_val, epochs = epochs)
    plot(history)
    
    model.save(modelname)
    print(f'Training model saved in ./model/{modelname}.h5')
    
    print('{:#^100}'.format(f" Testing Model with X_test "))
    model.multieval(X_test, y_test)

def model_eval(datadir= './data/csv', epochs = 100):
    repo = DataRepository(datadir=datadir)
    ny = repo.numClasses
    
    X_train, X_val, X_test, y_train, y_val, y_test, labels = repo.getDataTrain()
    
    acc_test = []
    runsec = []
    
    model = SignDetector()
    
    start_time = time.time()
    arch = 'zhanggru2'
    model.create_model(X_train, ny, arch)
    model.train(X_train, y_train, X_val, y_val, epochs = epochs)
    print('{:#^100}'.format(f" Testing Model with X_test "))
    acc = model.multieval(X_test, y_test)
    acc_test.append(acc)
    runsec.append(time.time() - start_time)
    
    start_time = time.time()
    arch = 'liulstm'
    model.create_model(X_train, ny, arch)
    model.train(X_train, y_train, X_val, y_val, epochs = epochs)
    print('{:#^100}'.format(f" Testing Model with X_test "))
    acc = model.multieval(X_test, y_test)
    acc_test.append(acc)
    runsec.append(time.time() - start_time)
    
    """start_time = time.time()
    arch = 'seqlstm2'
    model.create_model(X_train, ny, arch)
    model.train(X_train, y_train, X_val, y_val, epochs = epochs)
    print('{:#^100}'.format(f" Testing Model with X_test "))
    acc = model.multieval(X_test, y_test)
    acc_test.append(acc)
    runsec.append(time.time() - start_time)"""
    
    start_time = time.time()
    arch = 'seqlstm3'
    model.create_model(X_train, ny, arch)
    model.train(X_train, y_train, X_val, y_val, epochs = epochs)
    print('{:#^100}'.format(f" Testing Model with X_test "))
    acc = model.multieval(X_test, y_test)
    acc_test.append(acc)
    runsec.append(time.time() - start_time)

    """ arch = 'seqlstm'
    model.create_model(X_train, ny, arch)
    model.train(X_train, y_train, X_val, y_val, epochs = epochs)
    print('{:#^100}'.format(f" Testing Model with X_test "))
    acc = model.multieval(X_test, y_test)
    acc_test.append(acc) """
    
    """ arch = 'cnn'
    model.create_model(X_train, ny, arch)
    model.train(X_train, y_train, X_val, y_val, epochs = epochs)
    print('{:#^100}'.format(f" Testing Model with X_test "))
    acc = model.multieval(X_test, y_test)
    acc_test.append(acc)
    
    arch = 'connet'
    model.create_model(X_train, ny, arch)
    model.train(X_train, y_train, X_val, y_val, epochs = epochs)
    print('{:#^100}'.format(f" Testing Model with X_test "))
    acc = model.multieval(X_test, y_test)
    acc_test.append(acc) """
    
    return acc_test, runsec

def model_testing(model = None, datadir = './data/csv', epochs=1000):
    repo = DataRepository(datadir=datadir)
    
    X_train, X_val, X_test, y_train, y_val, y_test, labels = repo.getDataTrain()
    
    start_time = time.time()
    if model is None:
        ny = repo.numClasses
        model = SignDetector()
        arch = 'seqlstm3'
        model.create_model(X_train, ny, arch)
        model.train(X_train, y_train, X_val, y_val, epochs = epochs)
    
    print('{:#^100}'.format(f" Testing Model with X_test "))
    acc = model.multieval(X_test, y_test)
    runsec = time.time() - start_time
        
    return acc, runsec
    
    
    

if __name__=='__main__':
    # train_from_videos(epochs=1000)
    
    tes = []
    run_time = []
    
    start_time = time.time()
    # model = SignDetector('./model/tes.h5')
    for i in range(100):
        acc, runsec = model_testing()
        tes.append([acc, runsec])
    
    elapsed = time.time() - start_time
    df = pd.DataFrame(tes, columns=['acc', 'runtime'])
    df.to_excel('./testing3.xlsx', index=False)
    print(df.mean())
    print(elapsed)
    
    """ for i in range(100):
        acc, runsec = model_eval(epochs=1000)
        tes.append(acc)
        run_time.append(runsec)

    columns=['GRU (Zhang)', 'LSTM (Tao Liu)', 'LSTM2 (Proposed)','LSTM3 (Proposed)']
    
    df = pd.DataFrame(tes, columns = columns)
    df.to_excel('./runmodelfinal2.xlsx', index = False)
    print(df.mean())
    
    df = pd.DataFrame(run_time, columns=columns)
    df.to_excel('./runtimefinal2.xlsx', index = False)
    print(df.mean())  """
    
    