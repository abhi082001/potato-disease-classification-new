from fastapi import FastAPI, Path, File, UploadFile, Request, Form
from fastapi.templating import Jinja2Templates
import shutil
import os
from typing import Optional
from pydantic import BaseModel
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import json

app = FastAPI()

templates = Jinja2Templates(directory="frontend")

MODEL = tf.keras.models.load_model("./5")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
print(CLASS_NAMES)

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict_1(*,myfile: UploadFile = File(...), request: Request):
    
    img1 = await myfile.read()
    img = read_file_as_image(img1)
    print(np.shape(img))
    #img1 = np.resize(img, (256,256,3))
    img2 = tf.constant(img)
    img2 = tf.image.resize(img, [256,256]).numpy()
    print(np.shape(img2))
    #return 0
    img_batch = np.expand_dims(img2, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    confidence = round(confidence,3)
    confidence*=100
    #print(predicted_class)
    #print(assignment)
    #return {'predictedclass': predicted_class, "confidence": float(confidence)}
    return templates.TemplateResponse("pred.html", {"request": request, "pred_class": predicted_class, "confidence": float(confidence), "img": type(img1)})

@app.post("/test")
def test(l1):
    ans = l1*60
    return {"ans":float(ans)}

'''
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port = 8000)
'''
