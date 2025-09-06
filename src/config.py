# src/config.py
import os

# --- Main Project Directory ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- IP Camera URL ---
# Replace with your IP camera's URL
IP_CAMERA_URL = "http://192.168.1.7:8080/shot.jpg"

# --- Data Collection Parameters ---
NUM_SAMPLES = 100  # Number of samples to collect per person

# --- Data Directories ---
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RAW_DATA_DIR = os.path.join(DATASET_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATASET_DIR, "processed")

# --- Model and Cascade Classifier Paths ---
MODELS_DIR = os.path.join(BASE_DIR, "models")
CASCADE_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")
MODEL_PATH = os.path.join(MODELS_DIR, "face_recognition_model.h5")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# --- Image and Model Parameters ---
IMAGE_SIZE = (100, 100)
MODEL_INPUT_SHAPE = (100, 100, 1)

# --- Training Parameters ---
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0001