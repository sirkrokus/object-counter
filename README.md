# object-counter
A Python service that analyzes a video or a series of images and counts the number of moving objects such as people, animals, cars, etc.
The service requires dlib, numpy, opencv, imutils libraries. To use it the Anaconda platform recommended.
See **app.py** to start the service or use a code below
```
import detector

from event_counter \
    import ORIENTATION, LocationDescriptor, SimpleEventCounter, DIRECTION
from frame_provider import TestImageFrameProvider, TestVideoFrameProvider
from object_counter import ObjectCounter
from object_tracker import ObjectTracker

neuro_detector = detector.ObjectNeuroDetector("./mobilenet_ssd/MobileNetSSD_deploy.prototxt",
                                              "./mobilenet_ssd/MobileNetSSD_deploy.caffemodel",
                                              "person", 0.9)

evcnt = SimpleEventCounter()
location = LocationDescriptor(ORIENTATION.vertical)
object_tracker = ObjectTracker(evcnt, location, FRAME_SIZE / 2, 13, True)
frame_provider = TestVideoFrameProvider("./video/example_01_1040_1120_1u_2d.mp4")

objcnt = ObjectCounter(neuro_detector,
                       object_tracker,
                       frame_provider,
                       3, FRAME_SIZE,
                       null)
objcnt.run_counter()
print(f"STAT: {evcnt.stat}")
```

Recommended parameters for **people** counting
in ObjectNeuroDetector: 0.9 is a people recognition probability
```
"person", 0.9)
```

in ObjectTracker: 13 is number of frames when an object can be unrecognized
```
13, True)
```

in ObjectCounter: 3 is number of frames that will be missed (over jumped) to speed up the analysis
```
3, FRAME_SIZE,
```

Recommended parameters for **car** counting
in ObjectNeuroDetector: 0.5
```
"car", 0.5)
```

in ObjectTracker: 17
```
17, True)
```

in ObjectCounter: 2
```
2, FRAME_SIZE,
```

A test video with moving cars
```
frame_provider = TestVideoFrameProvider("./video/cars_2u_6d.mp4")
```
