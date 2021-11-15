from enum import Enum
from abc import ABCMeta, abstractmethod
import app_logger

DIRECTION = Enum("Direction", "up down left right")
# vertical - движутся вверх-вниз. horizontal - вправо-влево
ORIENTATION = Enum("Orientation", "vertical horizontal")


class LocationDescriptor:

    def __init__(self,
                 user_sid, project_sid, location_sid,
                 orientation: ORIENTATION):
        self.user_sid = user_sid
        self.project_sid = project_sid
        self.location_sid = location_sid
        self.orientation = orientation
        self.path = f"{user_sid}/{project_sid}/{location_sid}/"

    def __repr__(self):
        return f"LocationDescriptor. orientation={self.orientation}. path={self.path}"


class CountEvent:

    def __init__(self,
                 location: LocationDescriptor,  # для какой локации пришло событие
                 direction: DIRECTION,          # в какой стороне кадра скрылся объект
                 appear_time,                   # когда появился
                 disappear_time):               # когда пропал
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

