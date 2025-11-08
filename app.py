from flask import Flask, render_template, request, jsonify, url_for
import os
import face_recognition
from PIL import Image
import uuid
from werkzeug.utils import secure_filename
import base64
import io

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image(file_path):
    """Load an image file into a numpy array."""
    return face_recognition.load_image_file(file_path)

def compare_faces_web(image1_path, image2_path):
    """Compare two face images and return detailed results for web interface."""
    try:
        # Load the images
        image1 = load_image(image1_path)
        image2 = load_image(image2_path)

        # Encode the images
        image1_encoding = face_recognition.face_encodings(image1)
        image2_encoding = face_recognition.face_encodings(image2)

        if not image1_encoding and not image2_encoding:
            return {
                'success': False,
                'message': "Could not detect a face in both of the images.",
                'likelihood': 0
            }
        
        if not image1_encoding:
            return {
                'success': False,
                'message': "Could not detect a face in image 1.",
                'likelihood': 0
            }
        
        if not image2_encoding:
            return {
                'success': False,
                'message': "Could not detect a face in image 2.",
                'likelihood': 0
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
            'is_same_person': bool(results[0])
        }

    except Exception as e:
        return {
            'success': False,
            'message': f"Error processing images: {str(e)}",
            'likelihood': 0
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
    app.run(debug=True, host='0.0.0.0', port=5000)