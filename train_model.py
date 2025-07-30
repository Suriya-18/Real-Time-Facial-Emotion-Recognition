#!/usr/bin/env python3
"""
Train Emotion Recognition Model
This script trains the CNN model for facial emotion recognition.
"""

import argparse
import os
from model import train_model, load_trained_model
from utils import print_model_summary, save_training_history, check_model_files


def main():
    """
    Main function to train the emotion recognition model.
    """
    parser = argparse.ArgumentParser(description='Train Facial Emotion Recognition Model')
    parser.add_argument('--train-dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Path to testing data directory')
    parser.add_argument('--model-json', type=str, default='emotiondetector.json',
                        help='Path to save model architecture JSON file')
    parser.add_argument('--model-weights', type=str, default='emotiondetector.h5',
                        help='Path to save model weights file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--load-existing', action='store_true',
                        help='Load existing model and continue training')
    
    args = parser.parse_args()
    
    # Check if directories exist
    if not os.path.exists(args.train_dir):
        print(f"Error: Training directory not found at {args.train_dir}")
        return
    
    if not os.path.exists(args.test_dir):
        print(f"Error: Testing directory not found at {args.test_dir}")
        return
    
    print("="*60)
    print("FACIAL EMOTION RECOGNITION MODEL TRAINING")
    print("="*60)
    print(f"Training directory: {args.train_dir}")
    print(f"Testing directory: {args.test_dir}")
    print(f"Model JSON: {args.model_json}")
    print(f"Model weights: {args.model_weights}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("="*60)
    
    try:
        if args.load_existing and check_model_files(args.model_json, args.model_weights):
            print("Loading existing model for continued training...")
            model = load_trained_model(args.model_json, args.model_weights)
            print_model_summary(model)
            
            # Note: For continued training, you would need to modify the train_model function
            # to accept an existing model as a parameter
            print("Note: Continued training not implemented yet. Please train from scratch.")
            return
        
        # Train the model
        print("Starting model training...")
        model, label_encoder = train_model(
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            model_save_path=args.model_weights,
            json_save_path=args.model_json,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        print_model_summary(model)
        
        print("\nTraining completed successfully!")
        print(f"Model saved to: {args.model_weights}")
        print(f"Model architecture saved to: {args.model_json}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return


if __name__ == "__main__":
    main() 