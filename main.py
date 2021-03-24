import argparse

import mediapipe as mp
import cv2

import displayer
from camera import CameraStream
from video import VideoStream
from trainer import train_model_from_videos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sign to text: Command that parse a video stream and recognizes signs')
    parser.add_argument("-v", "--video", type=str, nargs='?', help='video input for testing')
    parser.add_argument("-m", "--model", type=str, nargs='?', help='pretrained model')
    parser.add_argument("-t", '--train', action="store_true", help='train or test')
    parser.add_argument("-n", "--no-evaluate", action="store_true")
    args = parser.parse_args()
    print(args)

    # init stream
    if args.video:
        stream = VideoStream(args.video)
    else:
        stream = CameraStream()

    # init components (hand landmark and pose landmark)
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    # launch action
    if args.no_evaluate:
        displayer.display_from_stream(stream, mp_pose, mp_hands)
    if args.train:
        train_model_from_videos()
    else:
        displayer.display_evaluate_from_stream(stream, mp_pose, mp_hands, args.model)
