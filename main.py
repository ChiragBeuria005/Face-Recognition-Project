# main.py
import argparse
import os
import sys

# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# It's important to import after modifying the path
from src.data_collection import collect_face_samples
from src.data_preprocessing import process_and_save_data
from src.model_training import train_model
from src.face_recognition import start_recognition

def main():
    parser = argparse.ArgumentParser(description="Face Recognition Project Pipeline")
    parser.add_argument(
        'action', 
        choices=['collect', 'preprocess', 'train', 'run'],
        help="Action to perform: 'collect' new face data, 'preprocess' the data, 'train' the model, or 'run' live recognition."
    )
    
    args = parser.parse_args()

    if args.action == 'collect':
        print("--- Starting Data Collection ---")
        collect_face_samples()
    elif args.action == 'preprocess':
        print("--- Starting Data Preprocessing ---")
        process_and_save_data()
    elif args.action == 'train':
        print("--- Starting Model Training ---")
        train_model()
    elif args.action == 'run':
        print("--- Starting Live Face Recognition ---")
        start_recognition()

if __name__ == "__main__":
    main()