import numpy as np
import cv2


class DetectedObject:

    def __init__(self, number, detected_position):
        self.number = number
        (startX, startY, endX, endY) = detected_position.astype("int")
        self.rectangle = (startX, startY, endX, endY)
        cx = int((startX + endX) / 2.0)
        cy = int((startY + endY) / 2.0)
        self.center = (cx, cy)


# распознавание объектов на картинке
class ObjectNeuroDetector:

    def __init__(self, path, model, class_name, min_confidence):
        print("[INFO] loading model '{0}' from '{1}'".format(model, path))
        self.net = cv2.dnn.readNetFromCaffe(path, model)
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        self.class_name = class_name
        self.min_confidence = min_confidence

    # сетка возвращает матрицу с распознанными объектами
    # return matrix of detected objects
    def detect(self, frame):
        (H, W) = frame.shape[:2]
        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        self.net.setInput(blob)
        return self.net.forward()

    # в матрице ищем объекты нужного класса с нужной вероятностью
    # и вычисляем координаты его прямоугольника на фрейме
    # frame_width, frame_height - frame size
    def find_object_positions(self, detections, frame_width, frame_height):
        # print("[INFO] detect '{0}' on a frame with minimum confidence {1}".format(class_name, min_confidence))
        positions = []  # DetectedObject

        # loop over the detections
        count = 0
        for i in np.arange(0, detections.shape[2]):
            if self.belongs_to_class(detections, i):
                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
                # (startX, startY, endX, endY) = box.astype("int")
                # print("BOX %d %d %d %d" % (startX, startY, endX, endY))
                count += 1
                positions.append(DetectedObject(count, box))

        return positions

    # принадлежит ли объект из матрицы под заданным индексом заданному классу
    # и какова вероятность этого
    def belongs_to_class(self, detections, idx):
        # extract the index of the class label from the detections list
        class_idx = int(detections[0, 0, idx, 1])
        # if the class label is not a person, ignore it
        # and
        # extract the confidence (i.e., probability) associated with the prediction
        # and
        # filter out weak detections by requiring a minimum confidence
        if detections[0, 0, idx, 2] < self.min_confidence or self.CLASSES[class_idx] != self.class_name:
            return False

        return True
