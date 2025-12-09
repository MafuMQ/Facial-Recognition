"""
Face Recognition Processing Module

This module handles face detection, comparison, gender detection, and age estimation.
Can be used standalone, by Flask app, or adapted for AWS Lambda.
"""

import cv2
import numpy as np
import face_recognition
from typing import Dict, Tuple, Optional


class FaceProcessor:
    """Handles all face recognition and analysis operations."""
    
    def __init__(self, gender_model_path: str = "gender_net.caffemodel",
                 gender_proto_path: str = "gender_deploy.prototxt",
                 age_model_path: str = "age_net.caffemodel",
                 age_proto_path: str = "age_deploy.prototxt"):
        """
        Initialize the FaceProcessor with gender and age detection models.
        
        Args:
            gender_model_path: Path to gender detection model
            gender_proto_path: Path to gender model prototxt
            age_model_path: Path to age detection model
            age_proto_path: Path to age model prototxt
        """
        self.GENDER_LIST = ['Male', 'Female']
        self.AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', 
                         '(38-43)', '(48-53)', '(60-100)']
        
        # Load gender detection model
        try:
            self.gender_net = cv2.dnn.readNetFromCaffe(gender_proto_path, gender_model_path)
        except Exception as e:
            self.gender_net = None
            print(f"Warning: Could not load gender model: {e}")
        
        # Load age detection model
        try:
            self.age_net = cv2.dnn.readNetFromCaffe(age_proto_path, age_model_path)
        except Exception as e:
            self.age_net = None
            print(f"Warning: Could not load age model: {e}")
    
    def detect_gender(self, face_img: np.ndarray) -> Tuple[str, float]:
        """
        Detect gender from a face image.
        
        Args:
            face_img: Face image as numpy array (BGR format)
            
        Returns:
            Tuple of (gender, confidence_percentage)
        """
        if self.gender_net is None:
            return "Unknown", 0.0
        
        try:
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), 
                                        [104, 117, 123], swapRB=False)
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender_idx = gender_preds[0].argmax()
            gender = self.GENDER_LIST[gender_idx]
            confidence = float(gender_preds[0][gender_idx]) * 100
            return gender, confidence
        except Exception as e:
            print(f"Error detecting gender: {e}")
            return "Unknown", 0.0
    
    def detect_age(self, face_img: np.ndarray) -> Tuple[str, float]:
        """
        Detect age range from a face image.
        
        Args:
            face_img: Face image as numpy array (BGR format)
            
        Returns:
            Tuple of (age_range, confidence_percentage)
        """
        if self.age_net is None:
            return "Unknown", 0.0
        
        try:
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), 
                                        [104, 117, 123], swapRB=False)
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age_idx = age_preds[0].argmax()
            age_range = self.AGE_LIST[age_idx]
            confidence = float(age_preds[0][age_idx]) * 100
            return age_range, confidence
        except Exception as e:
            print(f"Error detecting age: {e}")
            return "Unknown", 0.0
    
    def analyze_face(self, image: np.ndarray) -> Dict[str, any]:
        """
        Analyze a single image for face detection, gender, and age.
        
        Args:
            image: Image as numpy array (RGB format)
            
        Returns:
            Dictionary with face analysis results
        """
        result = {
            'face_detected': False,
            'gender': 'Unknown',
            'gender_confidence': 0.0,
            'age': 'Unknown',
            'age_confidence': 0.0,
            'face_encoding': None
        }
        
        # Detect faces
        face_encodings = face_recognition.face_encodings(image)
        face_locations = face_recognition.face_locations(image)
        
        if not face_encodings or not face_locations:
            return result
        
        result['face_detected'] = True
        result['face_encoding'] = face_encodings[0]
        
        # Extract face region for gender/age detection
        top, right, bottom, left = face_locations[0]
        face_img = image[top:bottom, left:right]
        
        if face_img.size == 0:
            return result
        
        # Convert to BGR for OpenCV
        face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
        
        # Detect gender and age
        gender, gender_conf = self.detect_gender(face_bgr)
        age, age_conf = self.detect_age(face_bgr)
        
        result['gender'] = gender
        result['gender_confidence'] = round(gender_conf, 1)
        result['age'] = age
        result['age_confidence'] = round(age_conf, 1)
        
        return result
    
    def compare_faces(self, image1_path: str, image2_path: str) -> Dict[str, any]:
        """
        Compare two face images and return detailed analysis.
        
        Args:
            image1_path: Path to first image file
            image2_path: Path to second image file
            
        Returns:
            Dictionary with comparison results including:
                - success: bool
                - message: str
                - likelihood: float (0-100)
                - is_same_person: bool
                - gender1, gender1_confidence, age1, age1_confidence
                - gender2, gender2_confidence, age2, age2_confidence
        """
        try:
            # Load images
            image1 = face_recognition.load_image_file(image1_path)
            image2 = face_recognition.load_image_file(image2_path)
            
            # Analyze both images
            analysis1 = self.analyze_face(image1)
            analysis2 = self.analyze_face(image2)
            
            # Build response
            result = {
                'success': True,
                'gender1': analysis1['gender'],
                'gender1_confidence': analysis1['gender_confidence'],
                'age1': analysis1['age'],
                'age1_confidence': analysis1['age_confidence'],
                'gender2': analysis2['gender'],
                'gender2_confidence': analysis2['gender_confidence'],
                'age2': analysis2['age'],
                'age2_confidence': analysis2['age_confidence'],
                'likelihood': 0,
                'is_same_person': False
            }
            
            # Check if faces were detected
            if not analysis1['face_detected'] and not analysis2['face_detected']:
                result['success'] = False
                result['message'] = "Could not detect a face in both of the images."
                return result
            
            if not analysis1['face_detected']:
                result['success'] = False
                result['message'] = "Could not detect a face in image 1."
                return result
            
            if not analysis2['face_detected']:
                result['success'] = False
                result['message'] = "Could not detect a face in image 2."
                return result
            
            # Compare face encodings
            face_distance = face_recognition.face_distance(
                [analysis1['face_encoding']], 
                analysis2['face_encoding']
            )[0]
            
            likelihood = round(100 * (1 - face_distance), 2)
            is_same = face_recognition.compare_faces(
                [analysis1['face_encoding']], 
                analysis2['face_encoding']
            )[0]
            
            result['likelihood'] = likelihood
            result['is_same_person'] = bool(is_same)
            
            if is_same:
                result['message'] = "These images are of the same person."
            else:
                result['message'] = "These images are not of the same person."
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Error processing images: {str(e)}",
                'likelihood': 0,
                'is_same_person': False,
                'gender1': 'Unknown',
                'gender1_confidence': 0.0,
                'age1': 'Unknown',
                'age1_confidence': 0.0,
                'gender2': 'Unknown',
                'gender2_confidence': 0.0,
                'age2': 'Unknown',
                'age2_confidence': 0.0
            }
