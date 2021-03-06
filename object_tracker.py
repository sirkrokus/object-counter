import vector_utils_2d as vu
import dlib
import cv2
import datetime
import app_logger
import utils as utils
from detector import DetectedObject
from event_counter \
    import LocationDescriptor, ORIENTATION, DIRECTION, AbstactEventCounter, CountEvent


class TrackableObject:

    def __init__(self, start_x, start_y, end_x, end_y, dlib_correlation_tracker):
        self.object_id = 0
        self.tracker = dlib_correlation_tracker
        self.rectangle = (start_x, start_y, end_x, end_y)
        cx = int((start_x + end_x) / 2.0)
        cy = int((start_y + end_y) / 2.0)
        self.center = (cx, cy)
        self.initial_center = (cx, cy)  # center of an object when it is appeared in an observable area
        self.final_center = None  # center of the object when it is hidden
        self.appear_time = datetime.datetime.now()
        self.disappear_time = None
        self.disappeared = False
        self.not_found_counter = 0  # counter how many times an object was not found (detected)
        self.log = app_logger.init()

    def consider_that_dissapear(self):
        self.not_found_counter += 1
        self.log.info(f"object #{self.object_id} is NOT found: {self.not_found_counter}")

    def set_disappear(self, force_dissapear: bool, max_missed_frames: int, on_dissapear_event_function):
        if self.disappeared:
            return
        # allow only a specified number of missed detections
        if not force_dissapear and self.not_found_counter <= max_missed_frames:
            return
        self.disappeared = True
        cx = int((self.rectangle[0] + self.rectangle[2]) / 2.0)
        cy = int((self.rectangle[1] + self.rectangle[3]) / 2.0)
        self.final_center = (cx, cy)
        self.disappear_time = datetime.datetime.now()
        self.log.info(f"object #{self.object_id} is disappear ({self.not_found_counter})! {self.rectangle}")
        on_dissapear_event_function(self)

    def set_rectangle(self, start_x, start_y, end_x, end_y):
        self.rectangle = (start_x, start_y, end_x, end_y)
        cx = int((start_x + end_x) / 2.0)
        cy = int((start_y + end_y) / 2.0)
        self.center = (cx, cy)

    def is_out_of_frame(self, frame_width, frame_height):
        b = self.center[0] < 0 or self.center[0] > frame_width or self.center[1] < 0 or self.center[1] > frame_height
        # self.log.debug(f"#{self.object_id} {self.rectangle} ?????? ???????????? ????????????? {b}")
        return b

    # distance to a point
    def distance_to_point(self, x, y):
        return vu.distance((self.center[0], self.center[1]), (x, y))

    # direction to a point - a rotation angle in degrees
    def angle_to_point(self, x, y):
        return vu.angle_clockwise(self.trackable_object.center, (x, y))

    def distance(self, trackable_object):
        return self.distance_to_point(trackable_object.center[0], trackable_object.center[1])

    def angle(self, trackable_object):
        return self.angle_to_point(trackable_object.center[0], trackable_object.center[1])

    def __repr__(self):
        return f"TrackableObject: id={self.object_id}. rect={self.rectangle}. center={self.center}. disappeared={self.disappeared}"


class ObjectTracker:

    def __init__(self,
                 event_counter: AbstactEventCounter,
                 location: LocationDescriptor,
                 max_centroid_distance,
                 max_missed_detections=9,
                 visualize: bool = False):
        self.next_object_id = 1
        self.max_centroid_distance = max_centroid_distance
        self.max_missed_detections = max_missed_detections
        self.trackable_objects = []
        self.event_counter = event_counter
        self.location = location
        self.frame_width = None
        self.frame_height = None
        self.visualize = visualize
        self.log = app_logger.init()

    def check_frame_size(self, frame):
        if self.frame_width is None or self.frame_height is None:
            (self.frame_height, self.frame_width) = frame.shape[:2]

    def on_dissapear_event(self, disappeared_object):
        dirn = None
        if self.location.orientation == ORIENTATION.horizontal:
            d = disappeared_object.initial_center[0] - disappeared_object.final_center[0]
            dirn = DIRECTION.right if d < 0 else DIRECTION.left

        if self.location.orientation == ORIENTATION.vertical:
            d = disappeared_object.initial_center[1] - disappeared_object.final_center[1]
            dirn = DIRECTION.down if d < 0 else DIRECTION.up

        event = CountEvent(self.location, dirn,
                           disappeared_object.appear_time,
                           disappeared_object.disappear_time)
        self.event_counter.count(event)

    # to update a frame of object create a new tracker based on detected object data
    def update_position(self, rgb_frame, tobj, detected_obj):
        tobj.tracker = self.create_tracker(rgb_frame, detected_obj)
        tobj.tracker.update(rgb_frame)
        tpos = tobj.tracker.get_position()
        (tstartX, tstartY, tendX, tendY) = utils.drect_to_int(tpos)
        tobj.set_rectangle(tstartX, tstartY, tendX, tendY)

    def create_trackable_object(self, rgb_frame, detected_obj):
        self.check_frame_size(rgb_frame)
        tracker = self.create_tracker(rgb_frame, detected_obj)
        (startX, startY, endX, endY) = detected_obj.rectangle
        tobj = TrackableObject(startX, startY, endX, endY, tracker)
        tobj.object_id = self.next_object_id
        self.log.debug(f"  new object is created {tobj}")
        self.next_object_id += 1
        return tobj

    def update_trackers(self, frame, rgb_frame):
        # self.log.debug("update trackers of all objects...")
        for tobj in self.trackable_objects:
            # self.log.debug(f"???????????????????????? ???????????? {tobj}")
            if tobj.tracker is not None and not tobj.disappeared:
                self.update_tracker(tobj, frame, rgb_frame)

    # update a object tracker and check if it is out of frame borders
    # ?????????? ???? ?????????????? ????????????
    def update_tracker(self, tobj, frame, rgb_frame):
        self.log.debug(f"update tracker for {tobj}")
        tpos = tobj.tracker.get_position()
        (tstartX, tstartY, tendX, tendY) = utils.drect_to_int(tpos)
        # self.log.debug("  old coords {}:{}, {}:{}".format(tstartX, tstartY, tendX, tendY))

        # self.on_dissapear_event(rgb_frame)
        tobj.tracker.update(rgb_frame)
        tpos = tobj.tracker.get_position()
        (tstartX, tstartY, tendX, tendY) = utils.drect_to_int(tpos)

        tobj.set_rectangle(tstartX, tstartY, tendX, tendY)
        # self.log.debug("  new coords {}:{}, {}:{}".format(tstartX, tstartY, tendX, tendY))

        # an object is out of frame borders
        if tobj.is_out_of_frame(self.frame_width, self.frame_height):
            tobj.set_disappear(True, self.max_missed_detections, self.on_dissapear_event)
            self.remove_all_disappeared()  # remove disappeared

        if self.visualize:
            self.draw_tracked_object(frame, tobj)

    def draw_tracked_object(self, frame, tobj: TrackableObject):
        # draw a tracked area on a frame
        (tstartX, tstartY, tendX, tendY) = tobj.rectangle
        cv2.rectangle(frame, (tstartX, tstartY), (tendX, tendY), (200, 0, 0))
        cv2.circle(frame, tobj.center, 3, (200, 0, 0))
        cv2.putText(frame, f"#{tobj.object_id}. {tstartX}:{tstartY}, {tendX}:{tendY}", (tstartX, tstartY - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 0, 0))

    def remove_all_disappeared(self):
        self.trackable_objects = [tobj for tobj in self.trackable_objects if not tobj.disappeared]

    # create a new tracker based on a detected object
    def create_tracker(self, rgb_frame, detected_obj):
        (startX, startY, endX, endY) = detected_obj.rectangle  # .astype("int")
        self.log.debug(f"create tracker based on a detected position {startX}:{startY}, {endX}:{endY}")
        tracker = dlib.correlation_tracker()
        rect = dlib.rectangle(startX, startY, endX, endY)
        tracker.start_track(rgb_frame, rect)
        return tracker

    def count(self):
        count = 0
        if len(self.trackable_objects) == 0:
            return count
        for tobj in self.trackable_objects:
            count += 0 if tobj.disappeared else 1
        return count

    def append_all(self, rgb_frame, detected_objects):
        for dobj in detected_objects:
            tobj = self.create_trackable_object(rgb_frame, dobj)
            self.trackable_objects.append(tobj)

    # calculate distances before all newly detected and tracked objects
    # and solve if some object is appered or disappeared
    def analyze_distances(self, rgb_frame, detected_objects):
        self.log.debug("calculate distances before centroids of all detected and tracked objects...")

        # calculate distances before all tracked and detected objects
        dist_list = []
        for dobj in detected_objects:
            for tobj in self.trackable_objects:
                if tobj.disappeared:
                    continue
                d = vu.distance(dobj.center, tobj.center)
                dist_list.append(Distance(d, tobj, dobj))

        new_tobjs = []  # newly tracked
        logged_d = []  # processed detected positions (new or upd)
        logged_t = []  # processed tracked (upd)

        def is_overdistanced(dist: Distance):
            (startX, startY, endX, endY) = dist.detected_object.rectangle
            dx = endX - startX
            dy = endY - startY
            mind = dx if dx < dy else dy  # short side
            # max possible distance before centroids it is a half of shortest side (of object's rectangle)
            # if the distance is bigger then the rectangles belong to different objects
            mind = mind / 2
            tf = dist.distance > mind
            msg = "big distance" if tf else "centroids are closely"
            self.log.debug(
                f"   {msg}. dx={dx}. dy={dy}. min_distance={mind}")
            return tf

        def is_d_not_used(dobj: DetectedObject):
            return not logged_d.__contains__(dobj.number)

        def is_t_not_used(tobj: TrackableObject):
            return not logged_t.__contains__(tobj.object_id)

        dist_list = sorted(dist_list, key=lambda x: x.distance)
        for d in dist_list:
            self.log.debug(f" : distance before Detected #{d.detected_object.number} and Tracked #{d.tracked_object.object_id} = {d.distance}")
            ovd = is_overdistanced(d)  # check every distance

            # if a detected object is far from some tracked object
            # and it is not yet counted in this checking loop
            # it means that this object is new and we need track it
            # create such object
            if ovd and is_d_not_used(d.detected_object):
                tobj = self.create_trackable_object(rgb_frame, d.detected_object)
                new_tobjs.append(tobj)
                logged_d.append(d.detected_object.number)

            # if a distance is small and detected object is not yet processed and tracker is not yet processed
            # update an area of tracked object with data of this newly detected object
            if not ovd and is_d_not_used(d.detected_object) and is_t_not_used(d.tracked_object):
                self.log.debug(
                    f"  tracked #{d.tracked_object.object_id}. update area {d.tracked_object.rectangle} -> {d.detected_object.rectangle}")
                self.update_position(rgb_frame, d.tracked_object, d.detected_object)
                logged_d.append(d.detected_object.number)
                logged_t.append(d.tracked_object.object_id)

        # these tracked objects were not detected
        # consider them disappeared
        for tobj in self.trackable_objects:
            if is_t_not_used(tobj):
                tobj.consider_that_dissapear()
                tobj.set_disappear(False, self.max_missed_detections, self.on_dissapear_event)

        self.remove_all_disappeared()

        # add all newly created to a common list
        for tobj in new_tobjs:
            self.trackable_objects.append(tobj)
            # print(f"{d}")

    # if a distance to a closest centroid is more then half of a distance from center to area's border
    # consider that a new object is appeared
    def is_newly_detected(self, dpos, min_distance):
        (startX, startY, endX, endY) = dpos.astype("int")
        dx = endX - startX
        dy = endY - startY
        d = dx if dx < dy else dy  # short side
        d = d / 2  # half of the short side
        tf = min_distance > d
        self.log.debug(f"is_newly_detected? {tf}. dx={dx}. dy={dy}. d={d}. min_distance={min_distance}")
        return tf

    # consider that all are disappeared
    def finalize_count(self):
        for tobj in self.trackable_objects:
            tobj.final_center = tobj.center
            self.on_dissapear_event(tobj)


class Distance:

    def __init__(self,
                 distance,
                 tracked_object: TrackableObject,
                 detected_object: DetectedObject):
        self.distance = distance
        self.tracked_object = tracked_object
        self.detected_object = detected_object

    def __repr__(self):
        return f"Dist={self.distance}, tobj #{self.tracked_object.object_id} {self.tracked_object.rectangle}, dobj #{self.detected_object.number} {self.detected_object.rectangle}"
