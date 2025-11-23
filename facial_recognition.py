# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:45:57 2024

@author: Mafu
"""

import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import face_recognition
import threading
import time
import cv2
import numpy as np

def load_image(file_path):
    """Load an image file into a numpy array."""
    return face_recognition.load_image_file(file_path)

# Gender detection setup
GENDER_MODEL = "gender_net.caffemodel"
GENDER_PROTO = "deploy_gender.prototxt"
GENDER_LIST = ['Male', 'Female']

try:
    gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
except Exception as e:
    gender_net = None
    print(f"Error loading gender model: {e}")

# Age detection setup
AGE_MODEL = "age_net.caffemodel"
AGE_PROTO = "age_deploy.prototxt"
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

try:
    age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
except Exception as e:
    age_net = None
    print(f"Error loading age model: {e}")

def detect_gender(face_img):
    """Detect gender from a face image (numpy array). Returns (gender, confidence)."""
    if gender_net is None:
        return "Unknown", 0.0
    # Preprocess face for gender model
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender_idx = gender_preds[0].argmax()
    gender = GENDER_LIST[gender_idx]
    confidence = gender_preds[0][gender_idx] * 100  # Convert to percentage
    return gender, confidence

def detect_age(face_img):
    """Detect age from a face image (numpy array). Returns (age_range, confidence)."""
    if age_net is None:
        return "Unknown", 0.0
    # Preprocess face for age model
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age_idx = age_preds[0].argmax()
    age_range = AGE_LIST[age_idx]
    confidence = age_preds[0][age_idx] * 100  # Convert to percentage
    return age_range, confidence

def display_image(image_label, file_path, gender_label=None, age_label=None):
    """Display the selected image in the specified label and show gender/age if labels provided."""
    image = Image.open(file_path)
    image.thumbnail((200, 200))  # Resize for display purposes
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo  # Keep a reference to avoid garbage collection
    # Detect gender and age and update labels
    if gender_label is not None or age_label is not None:
        img_np = np.array(image)
        # Convert to BGR for OpenCV
        if img_np.shape[-1] == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_np
        # Use face_recognition to find face locations
        faces = face_recognition.face_locations(img_np)
        if faces:
            top, right, bottom, left = faces[0]
            face_img = img_bgr[top:bottom, left:right]
            if face_img.size > 0:
                if gender_label is not None:
                    gender, gender_conf = detect_gender(face_img)
                    gender_text = f"{gender} ({gender_conf:.1f}%)"
                    gender_label.config(text=f"Gender: {gender_text}")
                if age_label is not None:
                    age, age_conf = detect_age(face_img)
                    age_text = f"{age} ({age_conf:.1f}%)"
                    age_label.config(text=f"Age: {age_text}")
            else:
                if gender_label is not None:
                    gender_label.config(text="Gender: Unknown")
                if age_label is not None:
                    age_label.config(text="Age: Unknown")
        else:
            if gender_label is not None:
                gender_label.config(text="Gender: No face detected")
            if age_label is not None:
                age_label.config(text="Age: No face detected")

def compare_faces(image1_path, image2_path):
    """Compare two face images and return if they are the same person or the likelihood."""
    if not image1_path or not image2_path:
        return "Please load both images before comparing."

    # Load the images
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)

    # Encode the images
    image1_encoding = face_recognition.face_encodings(image1)
    image2_encoding = face_recognition.face_encodings(image2)

    if not image1_encoding and not image2_encoding:
        return "Could not detect a face in both of the images."
    
    if not image1_encoding:
        return "Could not detect a face in image 1."
    
    if not image2_encoding:
        return "Could not detect a face in image 2."

    # Compare the faces
    results = face_recognition.compare_faces([image1_encoding[0]], image2_encoding[0])
    face_distance = face_recognition.face_distance([image1_encoding[0]], image2_encoding[0])[0]

    likelihood = f"Likelihood: {100 * (1 - face_distance):.2f}%"

    if results[0]:
        return f"These images are of the same person.\n{likelihood}"
    else:
        return f"These images are not of the same person.\n{likelihood}"

def open_file_dialog(image_label, image_var, gender_label, age_label):
    """Open a file dialog to select an image and display it, and show gender and age."""
    file_path = filedialog.askopenfilename()
    if file_path:
        image_var.set(file_path)
        display_image(image_label, file_path, gender_label, age_label)
        
def wrapper(a,b):
    result = compare_faces(a,b)
    result_label.config(text=result)

def compare_images():
    """Compare the two loaded images."""
    image1_path = image_a_var.get()
    image2_path = image_b_var.get()
    result_label.config(text="Comparing images, please wait...")

    thread = threading.Thread(target=wrapper, args=(image1_path, image2_path))
    thread.start()

def quit_app():
    root.destroy()

# Set up the main application window
root = tk.Tk()
root.title("Face Comparison Tool")

# Set a minimum size for the window to fit all UI elements
root.geometry("500x400")

# Variables to hold the image file paths
image_a_var = tk.StringVar()
image_b_var = tk.StringVar()

image_a_label = Label(root, text="Image A", width=200, height=200, relief="solid")
image_a_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
image_a_gender_label = Label(root, text="Gender: ", width=20)
image_a_gender_label.grid(row=0, column=0, padx=10, pady=(210,0), sticky="ew")
image_a_age_label = Label(root, text="Age: ", width=20)
image_a_age_label.grid(row=0, column=0, padx=10, pady=(235,0), sticky="ew")

image_b_label = Label(root, text="Image B", width=200, height=200, relief="solid")
image_b_label.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
image_b_gender_label = Label(root, text="Gender: ", width=20)
image_b_gender_label.grid(row=0, column=1, padx=10, pady=(210,0), sticky="ew")
image_b_age_label = Label(root, text="Age: ", width=20)
image_b_age_label.grid(row=0, column=1, padx=10, pady=(235,0), sticky="ew")

# Buttons to load images
load_image_a_button = tk.Button(root, text="Load Image A", command=lambda: open_file_dialog(image_a_label, image_a_var, image_a_gender_label, image_a_age_label))
load_image_a_button.grid(row=1, column=0, pady=10, sticky="ew")

load_image_b_button = tk.Button(root, text="Load Image B", command=lambda: open_file_dialog(image_b_label, image_b_var, image_b_gender_label, image_b_age_label))
load_image_b_button.grid(row=1, column=1, pady=10, sticky="ew")

# Button to compare images
compare_button = tk.Button(root, text="Compare Images", command=compare_images)
compare_button.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")

# Label to display the result
result_label = Label(root, text="", wraplength=400)
result_label.grid(row=3, column=0, columnspan=2, pady=10, sticky="ew")

# Quit button
quit_button = tk.Button(root, text="Quit", command=quit_app)
quit_button.grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")

# Adjust grid weights for resizing
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)

# Run the application
root.mainloop()

