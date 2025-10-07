from ultralytics import YOLO
import numpy as np
import cv2
from shapely.geometry import Polygon, box
from src.models import Detection, PredictionType, Segmentation
from src.config import get_settings

SETTINGS = get_settings()


def match_gun_bbox(segment: list[list[int]], bboxes: list[list[int]], max_distance: int = 10) -> list[int] | None:
    matched_box = None

    person_poly = Polygon(segment)

    min_distance = float("inf")

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        gun_box = box(x1, y1, x2, y2)

        distance = person_poly.distance(gun_box)

        if distance < min_distance:
            min_distance = distance
            matched_box = bbox

    if min_distance > max_distance:
        return None

    return matched_box


def annotate_detection(image_array: np.ndarray, detection: Detection) -> np.ndarray:
    ann_color = (255, 0, 0)
    annotated_img = image_array.copy()
    for label, conf, box in zip(detection.labels, detection.confidences, detection.boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_img, (x1, y1), (x2,y2), ann_color, 3)
        cv2.putText(
            annotated_img,
            f"{label}: {conf:.1f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            ann_color,
            2,
        )
    return annotated_img


def annotate_segmentation(image_array: np.ndarray, segmentation: Segmentation, draw_boxes: bool = True) -> np.ndarray:
    annotated_img = image_array.copy()

    for segment, bbox, label in zip(segmentation.polygons, segmentation.boxes, segmentation.labels):
        if label == "danger":
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        mask = np.zeros_like(annotated_img, dtype=np.uint8)
        pts = np.array(segment, dtype=np.int32)
        cv2.fillPoly(mask, [pts], color)
        annotated_img = cv2.addWeighted(annotated_img, 1, mask, 0.4, 0)

        if draw_boxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated_img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

    return annotated_img



class GunDetector:
    def __init__(self) -> None:
        print(f"loading od model: {SETTINGS.od_model_path}")
        self.od_model = YOLO(SETTINGS.od_model_path)
        print(f"loading seg model: {SETTINGS.seg_model_path}")
        self.seg_model = YOLO(SETTINGS.seg_model_path)

    def detect_guns(self, image_array: np.ndarray, threshold: float = 0.5):
        results = self.od_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        indexes = [
            i for i in range(len(labels)) if labels[i] in [3, 4]
        ]  # 0 = "person"
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in indexes
        ]
        confidences = [
            c for i, c in enumerate(results.boxes.conf.tolist()) if i in indexes
        ]
        labels_txt = [
            results.names[labels[i]] for i in indexes
        ]
        return Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels_txt,
            confidences=confidences,
        )
    
    def segment_people(self, image_array: np.ndarray, threshold: float = 0.5, max_distance: int = 10):
        results = self.seg_model(image_array, conf=threshold)[0]

        polygons = []
        boxes = []
        labels = []

        for i, cls_id in enumerate(results.boxes.cls.tolist()):
            label_name = results.names[int(cls_id)]
            if label_name != "person":
                continue

            if results.masks is None or results.masks.xy[i] is None:
                continue

            # Convertir cada punto del polígono a int
            segment = [[int(x), int(y)] for x, y in results.masks.xy[i].tolist()]
            if not segment:
                continue

            bbox = [int(v) for v in results.boxes.xyxy[i].tolist()]

            polygons.append(segment)
            boxes.append(bbox)

        guns_detection = self.detect_guns(image_array, threshold)
        gun_boxes = guns_detection.boxes if guns_detection.n_detections > 0 else []

        for segment, bbox in zip(polygons, boxes):
            matched_gun = match_gun_bbox(segment, gun_boxes, max_distance=max_distance)
            if matched_gun is not None:
                labels.append("danger")
            else:
                labels.append("safe")

        return Segmentation(
            pred_type=PredictionType.segmentation,
            n_detections=len(polygons),
            polygons=polygons,
            boxes=boxes,
            labels=labels,
        )