from fastapi import FastAPI, File, UploadFile
import uvicorn
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import io
import sys
from keras.models import load_model
from tensorflow.keras.models import  Sequential
import numpy as np
from PIL import Image
model_path = sys.argv[1] if len(sys.argv) > 1 else None
app = FastAPI()
model=None
def load_model_keras(path: str) -> Sequential:
    model = load_model(path) #Load the model from keras
    return model
def predict_digit(model:Sequential,data_point:list)->str:
    data_point = np.array(data_point, dtype=np.float32) / 255.0 #Data normalisation
    data_point = data_point.reshape(1, -1) #reshaping to None,784
    predict_digit = model.predict(data_point) #Get the predicitons
    return str(np.argmax(predict_digit))
#resizes image of any size to 28x28
def format_image(image: Image) -> Image: #This is part 2 of the problem statement
    image = image.resize((28, 28))
    return image
@app.get("/")
async def root():
    return {"message": "Hello World"} #Test for checking whether  the server is running or not( Not a part of the original problem statement)
@app.get("/load_model")
async def load_model_api(path: str):
    global model
    model = load_model_keras(r"C:\Users\sriva\Downloads\mnist_model2.h5") #Gets model from the local address where it is stored
@app.post('/predict',response_model=None)
async def predict_digit_api(upload_file: UploadFile = File(...)):
    global model
    if not model:
        return {"error": "Model is not loaded."}
    contents = await upload_file.read()
    
    # Open the image using PIL
    image = Image.open(io.BytesIO(contents)).convert('L') #Converts image to black and white
    # Resize the image to 28x28
    image=format_image(image)
    # Convert the image to a numpy array
    image_array = np.array(image)
    
    # Flatten the image array to a 1D array
    data_point = image_array.flatten().tolist()
    prediction = predict_digit(model, data_point)
    return {"predicted_digit": prediction}

uvicorn.run(app, host='0.0.0.0', port=8000) #This sets up host in 127.0.0.1:8000