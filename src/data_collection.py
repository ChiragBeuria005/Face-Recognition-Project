import cv2
import numpy as np
import urllib.request
import os
import socket
from . import config

def collect_face_samples():
    """
    Detects faces from the IP camera feed, collects a specified number of samples,
    and saves them to a person-specific directory. This version includes enhanced
    error handling and automatic directory creation.
    """
    # --- Step 1: Validate Haar Cascade Path ---
    if not os.path.exists(config.CASCADE_CLASSIFIER_PATH):
        print(f"Error: Cascade classifier not found at path: {config.CASCADE_CLASSIFIER_PATH}")
        print("Please ensure 'haarcascade_frontalface_default.xml' is in the 'models' directory.")
        return
        
    # --- New: Added try-except block to catch loading errors ---
    try:
        classifier = cv2.CascadeClassifier(config.CASCADE_CLASSIFIER_PATH)
    except cv2.error as e:
        print(f"Error loading CascadeClassifier from: {config.CASCADE_CLASSIFIER_PATH}")
        print(f"OpenCV Error: {e}")
        print("The XML file may be corrupted or in an invalid format. Please download a fresh copy.")
        return

    if classifier.empty():
        print(f"Error: Failed to load Haar Cascade classifier from {config.CASCADE_CLASSIFIER_PATH}.")
        print("The file might be corrupted or in an incorrect format.")
        return

    # --- Step 2: Get User Input and Create Directories ---
    name = input("Enter the name of the person: ")
    if not name:
        print("Name cannot be empty. Aborting.")
        return

    # Automatically create the main 'raw' data directory and the person-specific directory
    person_dir = os.path.join(config.RAW_DATA_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    
    print(f"Directory created for {name} at: {person_dir}")
    print("\nStarting data collection...")
    print("Look at the camera. The process will begin shortly.")
    print("Press 'q' in the camera window to quit at any time.")

    # --- Step 3: Collect Samples from Camera Feed ---
    samples_collected = 0
    while samples_collected < config.NUM_SAMPLES:
        try:
            # Fetch image from URL with a 5-second timeout
            with urllib.request.urlopen(config.IP_CAMERA_URL, timeout=5) as response:
                img_array = np.array(bytearray(response.read()), dtype=np.uint8)
            
            frame = cv2.imdecode(img_array, -1)
            
            if frame is None:
                print("Failed to decode frame from the camera. Retrying...")
                cv2.waitKey(1000) # Wait for a second before retrying
                continue

            # Detect faces
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = classifier.detectMultiScale(gray_frame, 1.3, 5)

            if len(faces) > 0:
                # Use the largest detected face
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                x, y, w, h = faces[0]

                # Crop the face and save the sample
                face_frame = frame[y:y+h, x:x+w]
                file_path = os.path.join(person_dir, f"{name}_{samples_collected}.jpg")
                cv2.imwrite(file_path, face_frame)

                samples_collected += 1
                print(f"Collected sample {samples_collected}/{config.NUM_SAMPLES}")
                
                # Draw feedback on the main frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Sample Collected!", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display progress on the frame
            progress_text = f"Progress: {samples_collected}/{config.NUM_SAMPLES}"
            cv2.putText(frame, progress_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("Collecting Samples... (Press 'q' to quit)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        except (urllib.error.URLError, socket.timeout) as e:
            print(f"Error: Could not connect to the camera at {config.IP_CAMERA_URL}.")
            print("Please check the URL in 'src/config.py' and your network connection.")
            print(f"Details: {e}")
            break # Exit the loop if the camera is not available
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Retrying...")
            continue
            
    cv2.destroyAllWindows()
    if samples_collected == config.NUM_SAMPLES:
        print("\nData collection complete!")
    else:
        print(f"\nData collection stopped. {samples_collected} samples were saved.")

