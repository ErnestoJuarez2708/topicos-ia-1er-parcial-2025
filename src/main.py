import io
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from fastapi.responses import Response
import numpy as np
from functools import cache
from PIL import Image, UnidentifiedImageError
from src.predictor import GunDetector, Detection, Segmentation, annotate_detection, annotate_segmentation
from src.models import Gun, GunType, Person, PersonType, PixelLocation
from src.config import get_settings

SETTINGS = get_settings()

app = FastAPI(title=SETTINGS.api_name, version=SETTINGS.revision)


@cache
def get_gun_detector() -> GunDetector:
    print("Creating model...")
    return GunDetector()

def detect_uploadfile(detector: GunDetector, file, threshold) -> tuple[Detection, np.ndarray]:
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )
    # convertir a una imagen de Pillow
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not suported"
        )
    # crear array de numpy
    img_array = np.array(img_obj)
    return detector.detect_guns(img_array, threshold), img_array

@app.post("/detect_people")
def detect_people(
    threshold: float = 0.5,
    max_distance: int = 10,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Segmentation:
    img_stream = io.BytesIO(file.file.read())
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Invalid image"
        )
    img_array = np.array(img_obj)

    segmentation = detector.segment_people(img_array, threshold, max_distance)
    return segmentation


@app.post("/annotate_people")
def annotate_people(
    threshold: float = 0.5,
    max_distance: int = 10,
    draw_boxes: bool = True,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    img_stream = io.BytesIO(file.file.read())
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Invalid image"
        )
    img_array = np.array(img_obj)

    segmentation = detector.segment_people(img_array, threshold, max_distance)
    annotated_img = annotate_segmentation(img_array, segmentation, draw_boxes=draw_boxes)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")


@app.post("/detect")
def detect_all(
    threshold: float = 0.5,
    max_distance: int = 10,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> dict:
    img_stream = io.BytesIO(file.file.read())
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Invalid image"
        )
    img_array = np.array(img_obj)

    detection = detector.detect_guns(img_array, threshold)
    segmentation = detector.segment_people(img_array, threshold, max_distance)

    return {
        "detection": detection.model_dump(),
        "segmentation": segmentation.model_dump(),
    }


@app.post("/annotate")
def annotate_all(
    threshold: float = 0.5,
    max_distance: int = 10,
    draw_boxes: bool = True,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    img_stream = io.BytesIO(file.file.read())
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Invalid image"
        )
    img_array = np.array(img_obj)

    detection = detector.detect_guns(img_array, threshold)
    segmentation = detector.segment_people(img_array, threshold, max_distance)

    annotated_img = annotate_detection(img_array, detection)
    annotated_img = annotate_segmentation(annotated_img, segmentation, draw_boxes=draw_boxes)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")


@app.post("/guns")
def get_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> list[Gun]:
    img_stream = io.BytesIO(file.file.read())
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Invalid image")
    img_array = np.array(img_obj)

    detection = detector.detect_guns(img_array, threshold)
    guns = []

    for label, bbox in zip(detection.labels, detection.boxes):
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        gun_type = GunType.pistol if "pistol" in label.lower() else GunType.rifle
        guns.append(Gun(gun_type=gun_type, location=PixelLocation(x=cx, y=cy)))

    return guns


@app.post("/people")
def get_people(
    threshold: float = 0.5,
    max_distance: int = 10,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> list[Person]:
    img_stream = io.BytesIO(file.file.read())
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Invalid image")
    img_array = np.array(img_obj)

    segmentation = detector.segment_people(img_array, threshold, max_distance)
    people = []

    for label, poly, bbox in zip(segmentation.labels, segmentation.polygons, segmentation.boxes):
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        pts = np.array(poly, dtype=np.int32)
        area = cv2.contourArea(pts)

        person_type = PersonType.danger if label == "danger" else PersonType.safe
        people.append(
            Person(
                person_type=person_type,
                location=PixelLocation(x=cx, y=cy),
                area=int(area),
            )
        )

    return people

@app.get("/model_info")
def get_model_info(detector: GunDetector = Depends(get_gun_detector)):
    return {
        "model_name": "Gun detector",
        "gun_detector_model": detector.od_model.model.__class__.__name__,
        "semantic_segmentation_model": detector.seg_model.model.__class__.__name__,
        "input_type": "image",
    }


@app.post("/detect_guns")
def detect_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Detection:
    results, _ = detect_uploadfile(detector, file, threshold)

    return results


@app.post("/annotate_guns")
def annotate_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    detection, img = detect_uploadfile(detector, file, threshold)
    annotated_img = annotate_detection(img, detection)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", port=8080, host="0.0.0.0")