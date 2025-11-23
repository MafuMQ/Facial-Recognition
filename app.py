from flask import Flask, render_template, request, jsonify, url_for
import os
import face_recognition
from PIL import Image
import uuid
from werkzeug.utils import secure_filename
import base64
import io
import cv2
import numpy as np
import argparse


app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Feature flags - can be set via command line arguments
ENABLE_GENDER_DETECTION = True
ENABLE_AGE_DETECTION = True

# Gender detection setup
GENDER_MODEL = "gender_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_LIST = ['Male', 'Female']

gender_net = None
if ENABLE_GENDER_DETECTION:
    try:
        gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
        print("Gender detection enabled")
    except Exception as e:
        print(f"Error loading gender model: {e}")
        print("Gender detection disabled")
else:
    print("Gender detection disabled")

# Age detection setup
AGE_MODEL = "age_net.caffemodel"
AGE_PROTO = "age_deploy.prototxt"
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

age_net = None
if ENABLE_AGE_DETECTION:
    try:
        age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
        print("Age detection enabled")
    except Exception as e:
        print(f"Error loading age model: {e}")
        print("Age detection disabled")
else:
    print("Age detection disabled")

def detect_gender(face_img):
    """Detect gender from a face image (numpy array). Returns (gender, confidence)."""
    if not ENABLE_GENDER_DETECTION or gender_net is None:
        return "Disabled", 0.0
    # Preprocess face for gender model
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender_idx = gender_preds[0].argmax()
    gender = GENDER_LIST[gender_idx]
    confidence = float(gender_preds[0][gender_idx]) * 100  # Convert to percentage and Python float
    return gender, confidence

def detect_age(face_img):
    """Detect age from a face image (numpy array). Returns (age_range, confidence)."""
    if not ENABLE_AGE_DETECTION or age_net is None:
        return "Disabled", 0.0
    # Preprocess face for age model
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age_idx = age_preds[0].argmax()
    age_range = AGE_LIST[age_idx]
    confidence = float(age_preds[0][age_idx]) * 100  # Convert to percentage and Python float
    return age_range, confidence

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image(file_path):
    """Load an image file into a numpy array."""
    return face_recognition.load_image_file(file_path)

def compare_faces_web(image1_path, image2_path):
    """Compare two face images and return detailed results for web interface, including gender."""
    try:
        # Load the images
        image1 = load_image(image1_path)
        image2 = load_image(image2_path)

        # Encode the images
        image1_encoding = face_recognition.face_encodings(image1)
        image2_encoding = face_recognition.face_encodings(image2)

        # Gender and age detection
        def get_gender_age(image, encodings):
            if not encodings:
                return "No face detected", 0.0, "No face detected", 0.0
            # Get first face location
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                return "No face detected", 0.0, "No face detected", 0.0
            top, right, bottom, left = face_locations[0]
            face_img = image[top:bottom, left:right]
            # Convert to BGR for OpenCV
            if face_img.size == 0:
                return "Unknown", 0.0, "Unknown", 0.0
            face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            gender, gender_conf = detect_gender(face_bgr)
            age, age_conf = detect_age(face_bgr)
            return gender, gender_conf, age, age_conf

        gender1, gender1_conf, age1, age1_conf = get_gender_age(image1, image1_encoding)
        gender2, gender2_conf, age2, age2_conf = get_gender_age(image2, image2_encoding)

        if not image1_encoding and not image2_encoding:
            return {
                'success': False,
                'message': "Could not detect a face in both of the images.",
                'likelihood': 0,
                'gender1': gender1,
                'gender1_confidence': round(gender1_conf, 1),
                'age1': age1,
                'age1_confidence': round(age1_conf, 1),
                'gender2': gender2,
                'gender2_confidence': round(gender2_conf, 1),
                'age2': age2,
                'age2_confidence': round(age2_conf, 1)
            }
        
        if not image1_encoding:
            return {
                'success': False,
                'message': "Could not detect a face in image 1.",
                'likelihood': 0,
                'gender1': gender1,
                'gender1_confidence': round(gender1_conf, 1),
                'age1': age1,
                'age1_confidence': round(age1_conf, 1),
                'gender2': gender2,
                'gender2_confidence': round(gender2_conf, 1),
                'age2': age2,
                'age2_confidence': round(age2_conf, 1)
            }
        
        if not image2_encoding:
            return {
                'success': False,
                'message': "Could not detect a face in image 2.",
                'likelihood': 0,
                'gender1': gender1,
                'gender1_confidence': round(gender1_conf, 1),
                'age1': age1,
                'age1_confidence': round(age1_conf, 1),
                'gender2': gender2,
                'gender2_confidence': round(gender2_conf, 1),
                'age2': age2,
                'age2_confidence': round(age2_conf, 1)
            }

        # Compare the faces
        results = face_recognition.compare_faces([image1_encoding[0]], image2_encoding[0])
        face_distance = face_recognition.face_distance([image1_encoding[0]], image2_encoding[0])[0]

        likelihood = round(100 * (1 - face_distance), 2)

        if results[0]:
            message = f"These images are of the same person."
        else:
            message = f"These images are not of the same person."

        return {
            'success': True,
            'message': message,
            'likelihood': likelihood,
            'is_same_person': bool(results[0]),
            'gender1': gender1,
            'gender1_confidence': round(gender1_conf, 1),
            'age1': age1,
            'age1_confidence': round(age1_conf, 1),
            'gender2': gender2,
            'gender2_confidence': round(gender2_conf, 1),
            'age2': age2,
            'age2_confidence': round(age2_conf, 1)
        }

    except Exception as e:
        return {
            'success': False,
            'message': f"Error processing images: {str(e)}",
            'likelihood': 0,
            'gender1': "Unknown",
            'gender1_confidence': 0.0,
            'age1': "Unknown",
            'age1_confidence': 0.0,
            'gender2': "Unknown",
            'gender2_confidence': 0.0,
            'age2': "Unknown",
            'age2_confidence': 0.0
        }

@app.route('/')
def index():
    """Main page of the web application."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and face comparison."""
    try:
        # Check if both files are present
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'success': False, 'message': 'Both images are required'})

        file1 = request.files['image1']
        file2 = request.files['image2']

        # Check if files are selected
        if file1.filename == '' or file2.filename == '':
            return jsonify({'success': False, 'message': 'Please select both images'})

        # Check file types
        if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
            return jsonify({'success': False, 'message': 'Invalid file type. Please use PNG, JPG, JPEG, GIF, BMP, or TIFF'})

        # Generate unique filenames
        filename1 = str(uuid.uuid4()) + '_' + secure_filename(file1.filename or 'upload1')
        filename2 = str(uuid.uuid4()) + '_' + secure_filename(file2.filename or 'upload2')

        # Save files
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

        file1.save(filepath1)
        file2.save(filepath2)

        # Compare faces
        result = compare_faces_web(filepath1, filepath2)

        # Add file URLs to result
        result['image1_url'] = url_for('static', filename=f'uploads/{filename1}')
        result['image2_url'] = url_for('static', filename=f'uploads/{filename2}')

        # Clean up: Remove uploaded files after processing
        try:
            os.remove(filepath1)
            os.remove(filepath2)
        except OSError:
            pass  # Ignore errors if files can't be deleted

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'})

@app.route('/clear', methods=['POST'])
def clear_uploads():
    """Clear uploaded images from the server."""
    try:
        # Remove all files in upload directory
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        return jsonify({'success': True, 'message': 'Upload directory cleared'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error clearing uploads: {str(e)}'})

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Face Recognition Web App')
    parser.add_argument('--no-gender', action='store_true', help='Disable gender detection')
    parser.add_argument('--no-age', action='store_true', help='Disable age detection')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    # Update feature flags based on arguments
    ENABLE_GENDER_DETECTION = not args.no_gender
    ENABLE_AGE_DETECTION = not args.no_age
    
    print(f"Starting Flask app on port {args.port}")
    print(f"Gender detection: {'Enabled' if ENABLE_GENDER_DETECTION else 'Disabled'}")
    print(f"Age detection: {'Enabled' if ENABLE_AGE_DETECTION else 'Disabled'}")
    
    app.run(debug=args.debug, host='0.0.0.0', port=args.port)