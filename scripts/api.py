from fastapi import FastAPI, Body
import numpy as np
import cv2
import base64
from modules.api.models import *
import gradio as gr
from io import BytesIO
from PIL import Image
import face_recognition as fr


def base64_to_image(base64_str):
    image = base64.b64decode(base64_str, altchars=None, validate=False)
    image = BytesIO(image)
    image = Image.open(image)
    return image


def face_recognition_api(_: gr.Blocks, app: FastAPI):
    @app.post("/face-recongnition")
    async def face_recognition(
            input_image: str = Body("", title='face-recongnition input image')
    ):
        image = base64_to_image(input_image)
        img = np.array(image)
        face_locations = fr.face_locations(img)
        if len(face_locations) == 0:
            return {"status": "Failed", "msg": "No real human face or no the full face in the input_image.", "data": ""}
        elif len(face_locations) > 1:
            return {"status": "Failed", "msg": "More than one face in the input_image.", "data": ""}
        else:
            top, right, bottom, left = face_locations[0]
            face = img[top:bottom, left:right]
            _, buffer = cv2.imencode('.png', face)
            return {"status": "Successful", "msg": "Successful", "data": base64.b64encode(buffer).decode()}


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(face_recognition_api)
except:
    pass
