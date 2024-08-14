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

def load_image(file_path):
    """Load an image file into a numpy array."""
    return face_recognition.load_image_file(file_path)

def display_image(image_label, file_path):
    """Display the selected image in the specified label."""
    image = Image.open(file_path)
    image.thumbnail((200, 200))  # Resize for display purposes
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo  # Keep a reference to avoid garbage collection

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

def open_file_dialog(image_label, image_var):
    """Open a file dialog to select an image and display it."""
    file_path = filedialog.askopenfilename()
    if file_path:
        image_var.set(file_path)
        display_image(image_label, file_path)
        
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

# Image display labels
image_a_label = Label(root, text="Image A", width=200, height=200, relief="solid")
image_a_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

image_b_label = Label(root, text="Image B", width=200, height=200, relief="solid")
image_b_label.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

# Buttons to load images
load_image_a_button = tk.Button(root, text="Load Image A", command=lambda: open_file_dialog(image_a_label, image_a_var))
load_image_a_button.grid(row=1, column=0, pady=10, sticky="ew")

load_image_b_button = tk.Button(root, text="Load Image B", command=lambda: open_file_dialog(image_b_label, image_b_var))
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

