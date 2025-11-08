# Facial Recognition Comparison Tool

A Python-based application that allows users to compare two face images and determine if they belong to the same person. The application provides a likelihood percentage and visual feedback through an intuitive interface.

**Available in two versions:**
- 🖥️ **Desktop GUI** (tkinter) - `facial_recognition.py`
- 🌐 **Web Application** (Flask) - `app.py`

## Features

- 🖼️ **Image Loading**: Easy file upload/selection with drag-and-drop support (web) or file dialog (desktop)
- 🔍 **Face Detection**: Automatic face detection in uploaded images
- 📊 **Similarity Analysis**: Calculates likelihood percentage of faces being the same person
- 🖥️ **User-Friendly Interface**: Clean GUI (tkinter) or modern web interface (Flask)
- ⚡ **Multi-threading**: Non-blocking comparison processing
- 🌐 **Web Accessibility**: Access from any device with a web browser (Flask version)
- 📱 **Responsive Design**: Mobile-friendly web interface

## Screenshots

### Desktop Version (tkinter)
The desktop application provides a simple interface with:
- Two image preview panels (Image A and Image B)
- Load buttons for each image
- Compare button to analyze similarity
- Results display with likelihood percentage

### Web Version (Flask)
The web application features:
- Modern, responsive design
- Drag-and-drop file upload
- Real-time image previews
- Progress indicators
- Mobile-friendly interface

## Prerequisites

Before running this application, ensure you have the following installed:

### Required Software
- **Python 3.7+** (Python 3.8 or higher recommended)
- **CMake** (required for dlib compilation)
- **Visual Studio Build Tools** or **Visual Studio** (for Windows users)

### CMake Installation
If CMake is not installed or not in your PATH:

1. **Windows**: Download from [cmake.org](https://cmake.org/download/)
   - During installation, select "Add CMake to system PATH"
   - Or manually add `C:\Program Files\CMake\bin` to your PATH

2. **Ubuntu/Debian**: 
   ```bash
   sudo apt install cmake
   ```

3. **macOS**: 
   ```bash
   brew install cmake
   ```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/MafuMQ/Facial-Recognition.git
cd Facial-Recognition
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: The installation of `dlib` may take several minutes as it needs to compile from source.

### 3. Verify Installation
Test that all modules can be imported:
```bash
python -c "import tkinter; import PIL; import face_recognition; print('All imports successful!')"
```

## Usage

### Desktop Version (tkinter)
```bash
python facial_recognition.py
```

### Web Version (Flask)
```bash
python app.py
```
Then open your web browser and navigate to `http://localhost:5000`

### How to Use Both Versions

#### Desktop Version:
1. **Launch the application** by running the Python script
2. **Load Image A**: Click "Load Image A" and select the first face image
3. **Load Image B**: Click "Load Image B" and select the second face image
4. **Compare**: Click "Compare Images" to analyze similarity
5. **View Results**: The application will display:
   - Whether the faces are from the same person
   - Likelihood percentage (confidence score)

#### Web Version:
1. **Launch the web server** by running `python app.py`
2. **Open your browser** to `http://localhost:5000`
3. **Upload Images**: 
   - Click on upload areas or drag-and-drop images
   - Images will be previewed immediately
4. **Compare**: Click "Compare Faces" to analyze similarity
5. **View Results**: The web interface will display:
   - Comparison result message
   - Confidence percentage with color coding
   - Side-by-side image comparison

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- And other formats supported by PIL

## Technical Details

### Dependencies
- **face_recognition**: Core facial recognition library
- **Pillow (PIL)**: Image processing and display
- **Flask**: Web framework (for web version)
- **tkinter**: GUI framework (included with Python, for desktop version)
- **numpy**: Mathematical operations (dependency of face_recognition)
- **dlib**: Computer vision library (dependency of face_recognition)

### How It Works
1. **Face Detection**: Uses HOG (Histogram of Oriented Gradients) for face detection
2. **Face Encoding**: Converts detected faces into 128-dimensional vectors
3. **Comparison**: Calculates Euclidean distance between face encodings
4. **Threshold**: Determines match based on distance threshold (typically 0.6)

### Performance Notes
- First-time face detection may be slower due to model loading
- Processing time depends on image size and complexity
- Multi-threading prevents UI freezing during processing

## Troubleshooting

### Common Issues

#### CMake Not Found Error
```
CMake is not installed on your system!
```
**Solution**: Install CMake and ensure it's in your system PATH (see Prerequisites section).

**Windows-Specific CMake PATH Fix**:
If CMake is installed but not recognized, you may need to add it to your PATH manually:

1. **Temporary Fix (Current Session Only)**:
   ```powershell
   $env:PATH += ";C:\Program Files\CMake\bin"
   ```

2. **Permanent Fix (Recommended)**:
   ```powershell
   # Add CMake to user PATH permanently
   $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
   $newPath = $currentPath + ";C:\Program Files\CMake\bin"
   [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
   ```

3. **Verify CMake is accessible**:
   ```powershell
   cmake --version
   ```

4. **Alternative Manual Method**:
   - Open Windows Settings → System → About → Advanced system settings
   - Click "Environment Variables"
   - Under "User variables", select "Path" and click "Edit"
   - Click "New" and add: `C:\Program Files\CMake\bin`
   - Click "OK" to save and restart your terminal

#### Visual Studio Build Tools Missing
```
Microsoft Visual C++ 14.0 is required
```
**Solution**: Install Visual Studio Build Tools or Visual Studio Community.

#### No Face Detected
```
Could not detect a face in image X
```
**Solution**: 
- Ensure the image contains a clear, front-facing face
- Try images with better lighting
- Face should be reasonably large in the image

#### Import Errors
```
ModuleNotFoundError: No module named 'face_recognition'
```
**Solution**: Run `pip install -r requirements.txt` to install dependencies.

### Getting Help
If you encounter issues:
1. Check that all prerequisites are installed
2. Verify your Python version is 3.7+
3. Ensure CMake is accessible: `cmake --version`
4. Try running the verification command from the Installation section

## Development

### Project Structure
```
Facial-Recognition/
├── facial_recognition.py    # Desktop GUI application (tkinter)
├── app.py                   # Web application (Flask)
├── templates/               # HTML templates for web version
│   └── index.html          # Main web interface
├── static/                  # Static files for web version
│   └── uploads/            # Temporary upload storage
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── LICENSE                 # License information
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built using the excellent [face_recognition](https://github.com/ageitgey/face_recognition) library by Adam Geitgey
- Face detection powered by [dlib](http://dlib.net/)
- GUI created with Python's built-in tkinter library

## Version History

- **v1.0.0**: Initial release with basic face comparison functionality

---

**Author**: Mafu  
**Created**: August 13, 2024  
**Last Updated**: November 8, 2025