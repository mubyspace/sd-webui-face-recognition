from fastapi import FastAPI, Body
import cv2
import insightface
import numpy as np
import base64
from modules.api.models import *
from modules.api import api
import gradio as gr
import os

xml_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "haarcascade_frontalface_default.xml")

def base64_to_cv2(base64_code):
    img_data = base64.b64decode(base64_code)
    img_array = np.fromstring(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    return img

def face_recognition_api(_: gr.Blocks, app: FastAPI):
	@app.post("/face-recongnition")
	async def face_recognition(
		input_image: str = Body("", title='face-recongnition input image')
	):
		img = base64_to_cv2(input_image)

		model = insightface.app.FaceAnalysis(name="buffalo_l")
		model.prepare(ctx_id=0, det_thresh=0.45, det_size=(640, 640))
		insightface_faces = model.get(img)

		face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		cv_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

		if len(insightface_faces) == 0 and len(cv_faces) == 0:
			return {"status": "Failed", "msg": "No faces in input_image"}
		elif len(insightface_faces) > 0 and len(cv_faces) == 0:
			return {"status": "Failed", "msg": "Please make sure that the input_image has the full face"}
		elif len(insightface_faces) > 1 and len(cv_faces) > 0:
			return {"status": "Failed", "msg": "There is more than one face in the input_image"}
		else:
			(x, y, w, h) = cv_faces[0]
			face = img[y:y+h, x:x+w]
			_, buffer = cv2.imencode('.png', face)
			return {"status": "Successed", "image": base64.b64encode(buffer).decode()}

try:
	import modules.script_callbacks as script_callbacks
	script_callbacks.on_app_started(face_recognition_api)
except:
	pass
