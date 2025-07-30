"""
Utility functions for Facial Emotion Recognition
This module contains common utility functions and configurations.
"""

import os
import numpy as np
from keras_preprocessing.image import load_img
import matplotlib.pyplot as plt


# Emotion labels
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Model configuration
MODEL_CONFIG = {
    'input_shape': (48, 48, 1),
    'num_classes': 7,
    'batch_size': 128,
    'epochs': 100
}

# File paths
DEFAULT_MODEL_JSON = 'emotiondetector.json'
DEFAULT_MODEL_WEIGHTS = 'emotiondetector.h5'


def extract_features_from_image(image_path):
    """
    Extract features from a single image file.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Preprocessed image features
    """
    img = load_img(image_path, color_mode='grayscale')
    feature = np.array(img)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


def display_image_with_prediction(image_path, predicted_emotion, true_emotion=None):
    """
    Display an image with its predicted emotion.
    
    Args:
        image_path (str): Path to the image file
        predicted_emotion (str): Predicted emotion label
        true_emotion (str, optional): True emotion label for comparison
    """
    img = load_img(image_path, color_mode='grayscale')
    img_array = np.array(img)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img_array, cmap='gray')
    
    title = f"Predicted: {predicted_emotion}"
    if true_emotion:
        title += f" | True: {true_emotion}"
    
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.show()


def check_model_files(model_json_path=None, model_weights_path=None):
    """
    Check if model files exist.
    
    Args:
        model_json_path (str, optional): Path to model JSON file
        model_weights_path (str, optional): Path to model weights file
        
    Returns:
        bool: True if both files exist, False otherwise
    """
    if model_json_path is None:
        model_json_path = DEFAULT_MODEL_JSON
    if model_weights_path is None:
        model_weights_path = DEFAULT_MODEL_WEIGHTS
    
    json_exists = os.path.exists(model_json_path)
    weights_exists = os.path.exists(model_weights_path)
    
    if not json_exists:
        print(f"Warning: Model JSON file not found at {model_json_path}")
    
    if not weights_exists:
        print(f"Warning: Model weights file not found at {model_weights_path}")
    
    return json_exists and weights_exists


def create_directory_if_not_exists(directory_path):
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path (str): Path to the directory to create
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")


def get_emotion_color(emotion):
    """
    Get color for displaying different emotions.
    
    Args:
        emotion (str): Emotion label
        
    Returns:
        tuple: BGR color tuple for OpenCV
    """
    color_map = {
        'angry': (0, 0, 255),      # Red
        'disgust': (0, 165, 255),  # Orange
        'fear': (255, 0, 255),     # Magenta
        'happy': (0, 255, 0),      # Green
        'neutral': (255, 255, 255), # White
        'sad': (255, 0, 0),        # Blue
        'surprise': (0, 255, 255)  # Yellow
    }
    
    return color_map.get(emotion.lower(), (255, 255, 255))


def print_model_summary(model):
    """
    Print a formatted model summary.
    
    Args:
        model: Keras model to summarize
    """
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    model.summary()
    print("="*50)


def save_training_history(history, filename='training_history.png'):
    """
    Save training history plots.
    
    Args:
        history: Training history object from model.fit()
        filename (str): Filename to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training history saved to {filename}")


if __name__ == "__main__":
    print("Utility module loaded successfully!")
    print(f"Available emotion labels: {EMOTION_LABELS}")
    print(f"Model configuration: {MODEL_CONFIG}") 