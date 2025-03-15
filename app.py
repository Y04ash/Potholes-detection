from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLO model
model_path = "pothole_detection_model.pt"  # Change this if needed
model = YOLO(model_path)

@app.route('/')
def home():
    return send_from_directory('.', 'index.html') 

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded/captured file
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)

    print(f"📂 Uploaded Image: {image_path}")

    # Perform pothole detection
    results = model(image_path)

    if not results:
        return jsonify({'error': 'No detections made'}), 500

    # Get annotated image
    result = results[0]  
    annotated_image = result.plot()  

    # Convert to BGR format before saving
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # Save the processed image
    output_filename = "output_" + file.filename
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    cv2.imwrite(output_path, annotated_image)

    # Return the path of the processed image
    return jsonify({'output_image': f"/uploads/{output_filename}"})

# Route to serve processed images
@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
