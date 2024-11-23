from flask import Flask, request, jsonify, render_template
import os
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Ensure the 'uploads' folder exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

def analyze_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    result = {label: str(prob) for (_, label, prob) in decoded_predictions}
    return result

@app.route('/')
def index():
    return render_template('/templates/index.html')  # Render the frontend HTML

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    # Save the file temporarily
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Call the image analysis function
    analysis_result = analyze_image(file_path)

    # Optionally, remove the temporary file
    os.remove(file_path)

    return jsonify(analysis_result)

if __name__ == '__main__':
    app.run(debug=True)
