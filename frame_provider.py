import cv2
from abc import ABCMeta, abstractmethod


class FrameProvider(metaclass=ABCMeta):

    @abstractmethod
    def next_frame(self):
        pass


class TestImageFrameProvider(FrameProvider):

    def __init__(self, path: str):
        self.path = path
        self.frame_index = 0

    def next_frame(self):
        self.frame_index += 1
        fname = f"{self.path}/{self.frame_index}.jpg"
        return cv2.imread(fname)


class TestVideoFrameProvider(FrameProvider):

    def __init__(self, path: str):
        self.path = path
        self.frame_index = 0
        self.video_stream = cv2.VideoCapture(f"{self.path}")

    def next_frame(self):
        self.frame_index += 1
        frame = self.video_stream.read()
        return frame[1]
