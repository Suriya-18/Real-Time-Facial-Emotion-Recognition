"""
Facial Emotion Recognition CNN Model
This module contains the CNN model definition and training functionality.
"""

import os
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras_preprocessing.image import load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def create_dataframe(directory):
    """
    Create a DataFrame containing image paths and their corresponding labels.
    
    Args:
        directory (str): Path to the directory containing emotion subdirectories
        
    Returns:
        tuple: (image_paths, labels) lists
    """
    image_paths = []
    labels = []
    
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            for imagename in os.listdir(label_dir):
                image_paths.append(os.path.join(label_dir, imagename))
                labels.append(label)
            print(f"{label} completed")
    
    return image_paths, labels


def extract_features(images):
    """
    Extract features from a list of images.
    
    Args:
        images (list): List of image file paths
        
    Returns:
        numpy.ndarray: Array of extracted features
    """
    features = []
    for image in tqdm(images, desc="Extracting features"):
        img = load_img(image, color_mode='grayscale')
        img = np.array(img)
        features.append(img)
    
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features


def create_emotion_model():
    """
    Create and return the CNN model for emotion recognition.
    
    Returns:
        tensorflow.keras.Model: Compiled CNN model
    """
    model = Sequential()
    
    # Convolutional layers
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    
    # Flattening
    model.add(Flatten())
    
    # Fully connected layers
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    
    # Output layer
    model.add(Dense(7, activation='softmax'))
    
    # Model compilation
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def prepare_data(train_dir, test_dir):
    """
    Prepare training and testing data from directories.
    
    Args:
        train_dir (str): Path to training data directory
        test_dir (str): Path to testing data directory
        
    Returns:
        tuple: (x_train, x_test, y_train, y_test, label_encoder)
    """
    print("Creating training data DataFrame...")
    train = pd.DataFrame()
    train['image'], train['label'] = create_dataframe(train_dir)
    
    print("Creating testing data DataFrame...")
    test = pd.DataFrame()
    test['image'], test['label'] = create_dataframe(test_dir)
    
    print("Extracting training features...")
    train_features = extract_features(train['image'])
    
    print("Extracting testing features...")
    test_features = extract_features(test['image'])
    
    # Normalize features
    x_train = train_features / 255.0
    x_test = test_features / 255.0
    
    # Encode labels
    le = LabelEncoder()
    le.fit(train['label'])
    y_train = le.transform(train['label'])
    y_test = le.transform(test['label'])
    y_train = to_categorical(y_train, num_classes=7)
    y_test = to_categorical(y_test, num_classes=7)
    
    return x_train, x_test, y_train, y_test, le


def train_model(train_dir, test_dir, model_save_path="emotiondetector.h5", 
                json_save_path="emotiondetector.json", epochs=100, batch_size=128):
    """
    Train the emotion recognition model.
    
    Args:
        train_dir (str): Path to training data directory
        test_dir (str): Path to testing data directory
        model_save_path (str): Path to save the trained model
        json_save_path (str): Path to save the model architecture
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        tensorflow.keras.Model: Trained model
    """
    # Prepare data
    x_train, x_test, y_train, y_test, le = prepare_data(train_dir, test_dir)
    
    # Create and train model
    model = create_emotion_model()
    
    print(f"Training model for {epochs} epochs...")
    model.fit(x=x_train, y=y_train, batch_size=batch_size, 
              epochs=epochs, validation_data=(x_test, y_test))
    
    # Save model
    model_json = model.to_json()
    with open(json_save_path, 'w') as json_file:
        json_file.write(model_json)
    model.save(model_save_path)
    
    print(f"Model saved to {model_save_path}")
    print(f"Model architecture saved to {json_save_path}")
    
    return model, le


def load_trained_model(model_json_path, model_weights_path):
    """
    Load a pre-trained model from saved files.
    
    Args:
        model_json_path (str): Path to the model architecture JSON file
        model_weights_path (str): Path to the model weights file
        
    Returns:
        tensorflow.keras.Model: Loaded model
    """
    from keras.models import model_from_json
    
    json_file = open(model_json_path, "r")
    model_json = json_file.read()
    json_file.close()
    
    model = model_from_json(model_json)
    model.load_weights(model_weights_path)
    
    return model


def predict_emotion(model, image_path, labels=None):
    """
    Predict emotion from an image file.
    
    Args:
        model: Trained emotion recognition model
        image_path (str): Path to the image file
        labels (list): List of emotion labels
        
    Returns:
        str: Predicted emotion label
    """
    if labels is None:
        labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    def extract_features_from_image(image_path):
        img = load_img(image_path, color_mode='grayscale')
        feature = np.array(img)
        feature = feature.reshape(1, 48, 48, 1)
        return feature / 255.0
    
    img = extract_features_from_image(image_path)
    pred = model.predict(img)
    pred_label = labels[pred.argmax()]
    
    return pred_label


if __name__ == "__main__":
    # Example usage
    TRAIN_DIR = r'C:\Face detection\archive (8)\images\images\train'
    TEST_DIR = r'C:\Face detection\archive (8)\images\images\test'
    
    # Train the model (uncomment to train)
    # model, le = train_model(TRAIN_DIR, TEST_DIR)
    
    # Load pre-trained model
    # model = load_trained_model("emotiondetector.json", "emotiondetector.h5")
    
    print("Model module loaded successfully!") 