from starlette.responses import HTMLResponse
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
import tensorflow as tf
import numpy as np

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL = tf.keras.models.load_model("C:/Users/Divya/Diabetic Retinopathy Classification/saved_model/3")
CLASS_NAMES = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]


@app.get("/")
async def index():
    with open("static/index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content, status_code=200)



def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(data.file))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(file)
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
