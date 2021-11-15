import imutils
import cv2
import detector
import app_logger
import copy

from frame_provider import FrameProvider
from object_tracker import ObjectTracker


class ObjectCounter:

    def __init__(self,
                 neuro_detector: detector.ObjectNeuroDetector,
                 object_tracker: ObjectTracker,
                 frame_provider: FrameProvider,
                 detect_frequency=6,
                 frame_size=500,
                 visualize=None):
        self.neuro_detector = neuro_detector
        self.object_tracker = object_tracker
        self.frame_provider = frame_provider
        self.detect_frequency = detect_frequency
        self.frame_size = frame_size
        self.frame_counter = 0
        self.visualize = visualize
        self.log = app_logger.init()

    def run_counter(self):

        while True:
            frame = self.frame_provider.next_frame()
            if frame is None:
                self.object_tracker.finalize_count()
                break

            self.frame_counter += 1
            frame_copy = self.process_frame(self.frame_counter, frame)

            if self.is_visualization_available():
                self.visualize(frame_copy)

        return self.frame_counter

    def is_visualization_available(self):
        return self.visualize is not None

    def process_frame(self, frame_counter, frame):
        # self.log.debug(f"=== FRAME {frame_counter} ===")

        frame = imutils.resize(frame, width=self.frame_size)
        frame_copy = copy.deepcopy(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # update coords of tracked objects
        self.object_tracker.update_trackers(frame_copy, rgb_frame)

        # detect objects on a frame
        if self.need_detect(frame_counter, self.detect_frequency):
            detected_objects = self.detect(frame)
            pos_len = len(detected_objects)
            self.log.debug(f"{pos_len} object(s) detected")

            # draw the found objects on the frame
            if self.is_visualization_available():
                self.draw_detected_object(detected_objects, frame_copy)

            self.sync_with_tracker(rgb_frame, detected_objects)

        return frame_copy

    def need_detect(self, frame_counter, freq):
        return frame_counter == 1 or frame_counter % freq == 0

    def detect(self, frame):
        detections = self.neuro_detector.detect(frame)
        (h, w) = frame.shape[:2]
        # [(startX, startY, endX, endY)]
        return self.neuro_detector.find_object_positions(detections, w, h)

    def draw_detected_object(self, detected_objects, frame_copy):
        for dobj in detected_objects:
            (startX, startY, endX, endY) = dobj.rectangle
            cv2.rectangle(frame_copy, (startX, startY), (endX, endY), (0, 0, 200))
            (cx, cy) = dobj.center
            cv2.circle(frame_copy, (cx, cy), 3, (0, 0, 200))
            cv2.putText(frame_copy, f"{startX}:{startY}, {endX}:{endY}", (startX, startY - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 200))

    # we need to understand what objects are new and what was already detected before
    def sync_with_tracker(self, rgb_frame, detected_objects):
        self.log.debug("sync with a tracker...")

        # if we don't have any objects add it all
        # else analyze distances before tracked and detected objects
        # and decide what became hidden or appear
        if self.object_tracker.count() == 0:
            self.object_tracker.append_all(rgb_frame, detected_objects)
        else:
            self.object_tracker.analyze_distances(rgb_frame, detected_objects)
