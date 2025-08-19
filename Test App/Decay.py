from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import boto3
import uuid
import os

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="C:/Users/Otinwa Ayomide/Downloads/Dissertation/Test App/resnet10_distilled_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels
class_names = ['Overripe', 'Ripe', 'Rotten', 'Unripe']

# AWS S3 config
S3_BUCKET = 'banananator'
S3_REGION = 'eu-north-1'  
s3 = boto3.client(
    's3',
    #will provide your AWS credentials here when needed
    aws_access_key_id='',
    aws_secret_access_key=''
)

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess uploaded image file."""
    img = Image.open(image).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction="No image uploaded", confidence=0)

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction="No file selected", confidence=0)

    # Preprocess image for model prediction
    input_data = preprocess_image(file)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_index = np.argmax(output_data)
    predicted_class = class_names[predicted_index]
    confidence = round(100 * output_data[predicted_index], 2)

    # Reset stream pointer and upload to S3
    file.stream.seek(0)
    s3_filename = f"{predicted_class}_{uuid.uuid4().hex}.jpg"
    s3.upload_fileobj(file, S3_BUCKET, s3_filename)

    s3_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_filename}"

    return render_template('index.html', prediction=predicted_class, confidence=confidence, image_url=s3_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
