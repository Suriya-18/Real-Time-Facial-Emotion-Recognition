"""
Real-time Facial Emotion Recognition
This module provides real-time emotion recognition using webcam feed.
"""

import cv2
import numpy as np
from keras.models import model_from_json
import argparse
import os


class EmotionRecognizer:
    """
    A class for real-time facial emotion recognition using webcam.
    """
    
    def __init__(self, model_json_path, model_weights_path, camera_id=0):
        """
        Initialize the emotion recognizer.
        
        Args:
            model_json_path (str): Path to the model architecture JSON file
            model_weights_path (str): Path to the model weights file
            camera_id (int): Camera device ID (default: 0 for default camera)
        """
        self.camera_id = camera_id
        self.labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 
                      4: 'neutral', 5: 'sad', 6: 'surprise'}
        
        # Load the pre-trained model
        self.model = self._load_model(model_json_path, model_weights_path)
        
        # Load the Haar cascade classifier for face detection
        haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_file)
        
        # Initialize webcam
        self.webcam = None
        
    def _load_model(self, model_json_path, model_weights_path):
        """
        Load the pre-trained emotion recognition model.
        
        Args:
            model_json_path (str): Path to the model architecture JSON file
            model_weights_path (str): Path to the model weights file
            
        Returns:
            tensorflow.keras.Model: Loaded model
        """
        try:
            json_file = open(model_json_path, "r")
            model_json = json_file.read()
            json_file.close()
            
            model = model_from_json(model_json)
            model.load_weights(model_weights_path)
            
            print(f"Model loaded successfully from {model_weights_path}")
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def extract_features(self, image):
        """
        Extract features from an image for emotion prediction.
        
        Args:
            image (numpy.ndarray): Input image array
            
        Returns:
            numpy.ndarray: Preprocessed image features
        """
        feature = np.array(image)
        feature = feature.reshape(1, 48, 48, 1)
        return feature / 255.0
    
    def predict_emotion(self, face_image):
        """
        Predict emotion from a face image.
        
        Args:
            face_image (numpy.ndarray): Face image array
            
        Returns:
            str: Predicted emotion label
        """
        if self.model is None:
            return "Model not loaded"
        
        # Resize face image to required input size
        face_image = cv2.resize(face_image, (48, 48))
        
        # Extract features
        img = self.extract_features(face_image)
        
        # Make prediction
        pred = self.model.predict(img)
        prediction_label = self.labels[pred.argmax()]
        
        return prediction_label
    
    def start_recognition(self):
        """
        Start real-time emotion recognition using webcam.
        """
        # Open webcam
        self.webcam = cv2.VideoCapture(self.camera_id)
        
        if not self.webcam.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Real-time Facial Emotion Recognition Started")
        print("Press 'Esc' to quit")
        
        while True:
            # Read a frame from the webcam
            ret, frame = self.webcam.read()
            
            if not ret:
                print("Error: Could not read frame from webcam")
                break
            
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the grayscale frame
            faces = self.face_cascade.detectMultiScale(frame, 1.3, 5)
            
            try:
                # For each detected face, perform facial emotion recognition
                for (x, y, w, h) in faces:
                    # Extract the region of interest (ROI) which contains the face
                    face_image = gray[y:y + h, x:x + w]
                    
                    # Draw a rectangle around the detected face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # Predict emotion
                    emotion = self.predict_emotion(face_image)
                    
                    # Display the predicted emotion label near the detected face
                    cv2.putText(frame, f'Emotion: {emotion}', (x - 10, y - 10),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                
                # Display the frame with annotations in real-time
                cv2.imshow("Real-time Facial Emotion Recognition", frame)
                
                # Break the loop if the 'Esc' key is pressed
                if cv2.waitKey(1) == 27:
                    break
                    
            except cv2.error as e:
                print(f"OpenCV error: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue
        
        self.stop_recognition()
    
    def stop_recognition(self):
        """
        Stop the emotion recognition and clean up resources.
        """
        if self.webcam is not None:
            self.webcam.release()
        cv2.destroyAllWindows()
        print("Emotion recognition stopped")


def main():
    """
    Main function to run real-time emotion recognition.
    """
    parser = argparse.ArgumentParser(description='Real-time Facial Emotion Recognition')
    parser.add_argument('--model-json', type=str, default='emotiondetector.json',
                        help='Path to model architecture JSON file')
    parser.add_argument('--model-weights', type=str, default='emotiondetector.h5',
                        help='Path to model weights file')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (default: 0)')
    
    args = parser.parse_args()
    
    # Check if model files exist
    if not os.path.exists(args.model_json):
        print(f"Error: Model JSON file not found at {args.model_json}")
        print("Please make sure you have trained the model first or provide correct paths.")
        return
    
    if not os.path.exists(args.model_weights):
        print(f"Error: Model weights file not found at {args.model_weights}")
        print("Please make sure you have trained the model first or provide correct paths.")
        return
    
    # Create and start emotion recognizer
    recognizer = EmotionRecognizer(args.model_json, args.model_weights, args.camera)
    
    try:
        recognizer.start_recognition()
    except KeyboardInterrupt:
        print("\nStopping emotion recognition...")
        recognizer.stop_recognition()


if __name__ == "__main__":
    main() 