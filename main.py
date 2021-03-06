import argparse

import mediapipe as mp
import cv2
import time

import displayer
from streamer import CameraStream, VideoStream
from trainer import train_from_videos

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sign to text: Command that parse a video stream and recognizes signs')
    parser.add_argument("-v", "--video", type=str, nargs='?', help='video input for testing')
    parser.add_argument("-m", "--model", type=str, nargs='?', help='pretrained model', default='./model/tes.h5')
    parser.add_argument("-t", '--train', action="store_true", help='train or test')
    parser.add_argument("-e", "--epoch", type=int, nargs='?', default=250)
    parser.add_argument("-s", '--save', type=str, nargs='?', help='model name', default='tes')
    parser.add_argument("-n", "--no-evaluate", action="store_true")
    args = parser.parse_args()
    start = time.time()
    
    # init stream
    if args.video:
        if './' not in args.video:
            vpath = './test/'+args.video
        else:
            vpath = args.video
        stream = VideoStream(vpath)
    else:
        stream = CameraStream()

    # init components (hand landmark and pose landmark)
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    # launch action
    if args.no_evaluate:
        displayer.display_from_stream(stream, mp_pose, mp_hands)
    if args.train:
        train_from_videos(modelname = args.save, epochs = args.epoch)
        pass
    else:
        if './' not in args.model:
            mpath = './model/' + args.model
        else:
            mpath = args.model
        displayer.display_evaluate_from_stream(stream, mp_pose, mp_hands, mpath)
    
    end = time.time()
    print(f'Runtime: {end-start} seconds')
