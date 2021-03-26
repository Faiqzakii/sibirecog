import os

import cv2
from pymediainfo import MediaInfo
from threading import Thread
from queue import Queue

class VideoStream:
    def __init__(self, video_path):
        # If not a video file, raise an exception to alert user about the issue
        self.video_path = video_path
        self.check_video()

        self.is_on = False
        self.cap = None
        

    def get_images(self):
        while self.cap.isOpened() and self.is_on:
            ret, frame = self.cap.read()

            self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                yield image
            else:
                self.close()
                break

    def open(self):
        self.is_on = True
        self.cap = cv2.VideoCapture(self.video_path)

    def close(self):
        self.is_on = False
        self.cap.release()

    def getwidth(self):
        return self.width

    def check_video(self):
        """
        Check if the video path is readable by parsing media info
        """
        if not os.path.isfile(self.video_path):
            raise Exception(f"The specified file doesn't exist: {self.video_path}")
        media_info = MediaInfo.parse(self.video_path)
        if len(media_info.video_tracks) == 0 or not media_info.video_tracks[0].track_type == "Video":
            raise Exception(f"The specified file is not a video: {self.video_path}")

class CameraStream:
    def __init__(self):
        self.is_on = False
        self.cap = None
        self.queue = Queue(maxsize=1)

    def get_images(self):
        while self.is_on:
            yield self.queue.get()

    def capture_image(self):
        self.cap = cv2.VideoCapture(0)
        while self.cap.isOpened() and self.is_on:
            success, image = self.cap.read()
            if not success:
                raise Exception("Bad reading of input camera")
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            if not self.queue.full():
                self.queue.put(image)

    def open(self):
        self.is_on = True
        thrd = Thread(target=self.capture_image, args=())
        thrd.daemon = True
        thrd.start()

    def close(self):
        self.is_on = False
        self.cap.release()
