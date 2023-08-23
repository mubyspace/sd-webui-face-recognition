from fastapi import FastAPI, Body
import numpy as np
import cv2
import base64
from modules.api.models import *
import gradio as gr
import insightface
from deepface import DeepFace


def base64_to_cv2(base64_code):
    img_data = base64.b64decode(base64_code)
    img_array = np.fromstring(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    return img


def insight_face(base64_code):
    img = base64_to_cv2(base64_code)
    model = insightface.app.FaceAnalysis(name="buffalo_l")
    model.prepare(ctx_id=0, det_thresh=0.45, det_size=(640, 640))
    faces = model.get(img)
    data = []
    msg = "Successful" if len(faces) > 0 else "Failed"
    for face in faces:
        x, y, w, h = face['bbox']
        x = int(min(max(x, 0), img.shape[1]))
        w = int(min(max(w + 1, 0), img.shape[1]))
        y = int(min(max(y, 0), img.shape[0]))
        h = int(min(max(h + 1, 0), img.shape[0]))
        face_area = img[y:h, x:w]
        _, buffer = cv2.imencode('.png', face_area)
        data.append(base64.b64encode(buffer).decode())
    return {"data": data, "msg": msg, "faces": len(faces)}


def deep_face(base64_code, detector):
    img = base64_to_cv2(base64_code)
    faces = DeepFace.extract_faces(img_path=img, detector_backend=detector, enforce_detection=False)
    data = []
    for face in faces:
        if face['confidence'] >= 0.9:
            x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], \
                         face['facial_area']['h']
            face_area = img[y:y + h, x:x + w]
            _, buffer = cv2.imencode('.png', face_area)
            data.append(base64.b64encode(buffer).decode())
    msg = "Successful" if len(data) > 0 else "Failed"
    return {"data": data, "msg": msg, "faces": len(data)}


def face_detection_api(_: gr.Blocks, app: FastAPI):
    @app.post("/face-detection")
    async def face_dection(
            input_image: str = Body("", title='face-detection input image'),
            detector: str = Body("", title="face-detection method")
    ):
        if detector == "insightface":
            return insight_face(input_image)
        else:
            return deep_face(input_image, detector)


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(face_detection_api)
except:
    pass
