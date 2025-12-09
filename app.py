from flask import Flask, render_template, request, jsonify, url_for
import os
import uuid
from werkzeug.utils import secure_filename
from face_processor import FaceProcessor


app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize face processor
face_processor = FaceProcessor()

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

        # Compare faces using face processor
        result = face_processor.compare_faces(filepath1, filepath2)

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