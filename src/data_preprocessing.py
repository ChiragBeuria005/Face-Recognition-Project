import os
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from . import config

def preprocess_image(image):
    """
    Applies consistent preprocessing to an image: resize, grayscale, histogram equalization.
    """
    image = cv2.resize(image, config.IMAGE_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    return image

def process_and_save_data():
    """
    Loads raw images, preprocesses them, splits into training/testing sets,
    encodes labels, and saves them as numpy arrays.
    """
    images = []
    labels = []

    print("Starting data preprocessing...")
    # Iterate through each person's folder in the raw data directory
    for person_name in os.listdir(config.RAW_DATA_DIR):
        person_dir = os.path.join(config.RAW_DATA_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        print(f"Processing images for: {person_name}")
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # Preprocess and store the image and its label
            images.append(preprocess_image(image))
            labels.append(person_name)

    if not images:
        print("No images found to process. Please run 'python main.py collect' first.")
        return

    # Convert lists to numpy arrays
    images = np.array(images, dtype='float32') / 255.0  # Normalize images
    images = np.expand_dims(images, axis=-1)  # Add channel dimension for CNN
    labels = np.array(labels)

    # Encode string labels to integers
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    
    # One-hot encode the integer labels
    labels_one_hot = to_categorical(labels_encoded)
    
    # --- New: Split data into training and testing sets (80% train, 20% test) ---
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels_one_hot, test_size=0.2, random_state=42, stratify=labels_one_hot
    )

    # Ensure the processed data and models directories exist
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Save the processed data splits and the label encoder
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'y_test.npy'), y_test)
        
    with open(config.ENCODER_PATH, 'wb') as f:
        pickle.dump(encoder, f)

    print("\nPreprocessing complete.")
    print(f"Total images processed: {len(images)}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Classes found: {list(encoder.classes_)}")
