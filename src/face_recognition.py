# src/face_recognition.py
import cv2
import urllib.request
import numpy as np
import pickle
from keras.models import load_model
from . import config

def start_recognition():
    """
    Starts live face recognition using the trained model and IP camera.
    """
    print("Loading trained model and resources...")
    try:
        model = load_model(config.MODEL_PATH)
        with open(config.ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)
        classifier = cv2.CascadeClassifier(config.CASCADE_CLASSIFIER_PATH)
    except Exception as e:
        print(f"Error loading resources: {e}")
        print("Please ensure the model, encoder, and cascade classifier exist.")
        return

    print("Starting live recognition. Press 'q' to quit.")

    while True:
        try:
            # Fetch image from URL
            with urllib.request.urlopen(config.IP_CAMERA_URL) as response:
                img_array = np.array(bytearray(response.read()), dtype=np.uint8)
            
            frame = cv2.imdecode(img_array, -1)
            if frame is None:
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = classifier.detectMultiScale(gray_frame, 1.3, 5)

            for (x, y, w, h) in faces:
                # Crop and preprocess the face for prediction
                face_roi = gray_frame[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, config.IMAGE_SIZE)
                face_roi = cv2.equalizeHist(face_roi)
                face_input = face_roi.astype('float32') / 255.0
                face_input = np.expand_dims(face_input, axis=0)
                face_input = np.expand_dims(face_input, axis=-1)

                # Predict the label
                prediction = model.predict(face_input)
                pred_index = np.argmax(prediction)
                confidence = prediction[0][pred_index]
                
                # Get the name from the label encoder
                pred_name = encoder.inverse_transform([pred_index])[0]

                # Display the result
                label_text = f"{pred_name} ({confidence:.2f})"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("Face Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
            break
            
    cv2.destroyAllWindows()