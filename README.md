# Real-Time Facial Emotion Recognition

A real-time facial emotion recognition system built with OpenCV and a Convolutional Neural Network (CNN). This project can detect and classify facial emotions such as **happiness, sadness, anger, surprise, fear, disgust, and neutrality** from live video streams.

## 🚀 Features

- **Real-time face detection** using OpenCV's Haar Cascade classifier
- **Emotion classification** with a pre-trained CNN model
- **Live webcam integration** for real-time emotion recognition
- **Modular architecture** with separate training and inference modules
- **Enhanced preprocessing** including grayscale conversion, resizing, and normalization
- **User-friendly interface** with visual feedback and emotion labels


### Dataset Acquisition:
The FER2013 dataset can be downloaded from:
- **Kaggle**: [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- **Official Source**: [ICML 2013 Workshop](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)


```
## 📁 Project Structure

```
Real-Time Facial Emotion Recognition/
├── model.py                          # CNN model definition and training
├── realtime_emotion_recognition.py   # Real-time emotion recognition
├── utils.py                          # Utility functions and configurations
├── train_model.py                    # Standalone training script
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
├── EMOTIONRECOGNITIONCNNMODEL.ipynb  # Original training notebook
└── EMOTIONRECOGNITIONCVFILE.ipynb    # Original real-time notebook
```

## 🛠️ Installation

### Prerequisites

- Python 3.7 or higher
- Webcam for real-time recognition
- GPU (optional, for faster training)


## 🎯 Usage

### 1. Training the Model

To train your own emotion recognition model:

```bash
python train_model.py --train-dir /path/to/train/data --test-dir /path/to/test/data
```

**Optional parameters:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 128)
- `--model-json`: Path to save model architecture (default: emotiondetector.json)
- `--model-weights`: Path to save model weights (default: emotiondetector.h5)

**Example:**
```bash
python train_model.py \
    --train-dir "C:\Face detection\archive (8)\images\images\train" \
    --test-dir "C:\Face detection\archive (8)\images\images\test" \
    --epochs 50 \
    --batch-size 64
```

### 2. Real-Time Emotion Recognition

To run real-time emotion recognition:

```bash
python realtime_emotion_recognition.py
```

**Optional parameters:**
- `--model-json`: Path to model architecture file (default: emotiondetector.json)
- `--model-weights`: Path to model weights file (default: emotiondetector.h5)
- `--camera`: Camera device ID (default: 0)

**Example:**
```bash
python realtime_emotion_recognition.py \
    --model-json emotiondetector.json \
    --model-weights emotiondetector.h5 \
    --camera 0
```

### 3. Using the Modules

You can also use the modules programmatically:

```python
from model import load_trained_model, predict_emotion
from realtime_emotion_recognition import EmotionRecognizer

# Load a trained model
model = load_trained_model('emotiondetector.json', 'emotiondetector.h5')

# Predict emotion from an image
emotion = predict_emotion(model, 'path/to/image.jpg')
print(f"Predicted emotion: {emotion}")

# Start real-time recognition
recognizer = EmotionRecognizer('emotiondetector.json', 'emotiondetector.h5')
recognizer.start_recognition()
```

## 📊 Dataset

This project uses the **FER2013** (Facial Expression Recognition 2013) dataset, which is one of the most widely used datasets for facial emotion recognition research.

### Dataset Statistics:
- **Total Images**: 35,887 grayscale images
- **Image Size**: 48x48 pixels
- **Training Set**: 28,821 images
- **Testing Set**: 7,066 images
- **Emotion Classes**: 7 (angry, disgust, fear, happy, neutral, sad, surprise)

### Dataset Structure:
```
dataset/
├── train/
│   ├── angry/      # Training angry emotion images
│   ├── disgust/    # Training disgust emotion images
│   ├── fear/       # Training fear emotion images
│   ├── happy/      # Training happy emotion images
│   ├── neutral/    # Training neutral emotion images
│   ├── sad/        # Training sad emotion images
│   └── surprise/   # Training surprise emotion images
└── test/
    ├── angry/      # Testing angry emotion images
    ├── disgust/    # Testing disgust emotion images
    ├── fear/       # Testing fear emotion images
    ├── happy/      # Testing happy emotion images
    ├── neutral/    # Testing neutral emotion images
    ├── sad/        # Testing sad emotion images
    └── surprise/   # Testing surprise emotion images
```

### Dataset Acquisition:
The FER2013 dataset can be downloaded from:
- **Kaggle**: [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- **Official Source**: [ICML 2013 Workshop](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)

### Dataset Citation:
If you use this dataset in your research, please cite:
```
Goodfellow, I. J., et al. "Challenges in representation learning: A report on three machine learning contests." Neural Networks 64 (2015): 59-63.
```

## 🏗️ Model Architecture

The CNN model consists of:

- **4 Convolutional layers** with ReLU activation and MaxPooling
- **Dropout layers** (0.4) for regularization
- **2 Dense layers** with ReLU activation
- **Output layer** with softmax activation for 7 emotion classes
- **Input shape**: (48, 48, 1) grayscale images

## 🎭 Supported Emotions

The model can recognize 7 different emotions:

1. **Angry** 😠
2. **Disgust** 🤢
3. **Fear** 😨
4. **Happy** 😊
5. **Neutral** 😐
6. **Sad** 😢
7. **Surprise** 😲

## 📈 Performance

- **Real-time processing** with webcam feed
- **Face detection** using Haar Cascade classifier
- **Emotion classification** with CNN model
- **Preprocessing**: Grayscale conversion, resizing to 48x48, normalization

## 🔧 Configuration

Key configurations can be modified in `utils.py`:

```python
# Emotion labels
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Model configuration
MODEL_CONFIG = {
    'input_shape': (48, 48, 1),
    'num_classes': 7,
    'batch_size': 128,
    'epochs': 100
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Webcam not working:**
   - Check if webcam is connected and accessible
   - Try different camera IDs (0, 1, 2, etc.)
   - Ensure no other application is using the webcam

2. **Model files not found:**
   - Make sure you have trained the model first
   - Check file paths in the command line arguments
   - Verify that `emotiondetector.json` and `emotiondetector.h5` exist

3. **Dependencies issues:**
   - Update pip: `pip install --upgrade pip`
   - Install dependencies individually if needed
   - Use virtual environment for isolation

4. **Performance issues:**
   - Reduce image resolution in preprocessing
   - Use GPU acceleration if available
   - Adjust batch size for training

### Error Messages

- **"Could not open webcam"**: Check camera connection and permissions
- **"Model not loaded"**: Verify model file paths and integrity
- **"OpenCV error"**: Update OpenCV or check image processing pipeline
