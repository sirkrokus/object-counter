import detector
import cv2

from event_counter \
    import ORIENTATION, LocationDescriptor, SimpleEventCounter, DIRECTION
from frame_provider import TestImageFrameProvider, TestVideoFrameProvider
from object_counter import ObjectCounter
from object_tracker import ObjectTracker

FRAME_SIZE = 600


def test_count():
    c = SimpleEventCounter()
    c.stat[DIRECTION.up] += 1
    c.stat[DIRECTION.down] += 1
    c.stat[DIRECTION.left] += 2
    c.stat[DIRECTION.right] += 2
    print(f"stat={c.stat}")


def visualize_in_frame(frame_copy):
    cv2.imshow("Frame", frame_copy)
    # cv2.waitKey(0)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        return False
    return True


def main():
    neuro_detector = detector.ObjectNeuroDetector("./mobilenet_ssd/MobileNetSSD_deploy.prototxt",
                                                  "./mobilenet_ssd/MobileNetSSD_deploy.caffemodel",
                                                  # "car", 0.5) # suggested params: max_missed_detections=17, detect_frequency=2
                                                  "person", 0.9) # suggested params: max_missed_detections=13, detect_frequency=3
                                                  # "aeroplane", 0.6)

    evcnt = SimpleEventCounter()
    location = LocationDescriptor(ORIENTATION.vertical)
    object_tracker = ObjectTracker(evcnt, location, FRAME_SIZE / 2, 13, True)
    # frame_provider = TestImageFrameProvider("./frames/test_queue_1")
    frame_provider = TestVideoFrameProvider("./video/example_01_1040_1120_1u_2d.mp4")
    # frame_provider = TestVideoFrameProvider("./video/cars_2u_6d.mp4")

    objcnt = ObjectCounter(neuro_detector,
                           object_tracker,
                           frame_provider,
                           3, FRAME_SIZE,
                           visualize_in_frame)
    objcnt.run_counter()
    print(f"STAT: {evcnt.stat}")


main()
# test_count()
