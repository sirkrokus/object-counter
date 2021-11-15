from enum import Enum
from abc import ABCMeta, abstractmethod
import app_logger

DIRECTION = Enum("Direction", "up down left right")
# vertical - objects are moving up and down. horizontal - left to right and vice versa
ORIENTATION = Enum("Orientation", "vertical horizontal")


class LocationDescriptor:

    def __init__(self,
                 orientation: ORIENTATION):
        self.orientation = orientation

    def __repr__(self):
        return f"LocationDescriptor. orientation={self.orientation}"


class CountEvent:

    def __init__(self,
                 location: LocationDescriptor,  # what location an event went for
                 direction: DIRECTION,          # what side of frame an object is hidden
                 appear_time,                   # when it is apperead
                 disappear_time):               # when it is hidden
        self.location = location
        self.direction = direction
        self.appear_time = appear_time
        self.disappear_time = disappear_time

    def __repr__(self):
        return f"CountEvent. direction=[{self.direction}]. loc={self.location}. time={self.appear_time}/{self.disappear_time}"


class AbstactEventCounter(metaclass=ABCMeta):

    @abstractmethod
    def count(self, event: CountEvent):
        pass


class MongoEventCounter(AbstactEventCounter):

    def count(self, event: CountEvent):
        pass


class SimpleEventCounter(AbstactEventCounter):

    def __init__(self):
        self.stat = {
            DIRECTION.up: 0,
            DIRECTION.down: 0,
            DIRECTION.left: 0,
            DIRECTION.right: 0
        }
        self.log = app_logger.init()

    def count(self, event: CountEvent):
        self.log.debug(f"{event}.")
        self.stat[event.direction] += 1

