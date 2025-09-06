Real-Time Face Recognition with IP Camera
This project is a complete pipeline for building, training, and deploying a real-time face recognition system using a Convolutional Neural Network (CNN). It captures live video from a smartphone or IP camera, detects faces, and identifies individuals it has been trained to recognize.

Features
Modular & Configurable: All settings (camera URL, file paths, training parameters) are centralized in src/config.py.

Data Augmentation: Artificially expands the dataset during training to build a more robust and accurate model, preventing overfitting.

Step-by-Step Pipeline: Simple command-line actions (collect, preprocess, train, run) to manage the workflow.

Live Recognition: Uses an IP camera stream (e.g., from a smartphone) for real-time face identification.

Detailed Evaluation: Automatically generates a classification report and a confusion matrix after training to show model performance.

Project Structure
face_recognition_project/
│
├── dataset/              # Stores raw and processed images
│   ├── raw/
│   └── processed/
│
├── models/               # Stores the Haar Cascade, trained model, and label encoder
│
├── src/                  # All source code
│   ├── config.py
│   ├── data_collection.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── face_recognition.py
│
├── main.py               # Main script to run all commands
├── requirements.txt      # Project dependencies
└── README.md             # This file

Setup Instructions
Step 1: Clone the Repository
Clone this repository to your local machine.

git clone <your-repository-url>
cd face_recognition_project

Step 2: Create a Virtual Environment (Recommended)
This keeps your project dependencies isolated.

python -m venv venv
venv\Scripts\activate  # On Windows

Step 3: Install Dependencies
Install all the required libraries from the requirements.txt file.

pip install -r requirements.txt

Step 4: Download the Face Detection Model
The program uses a Haar Cascade classifier to detect where faces are in the frame.

Create the models folder if it doesn't exist.

Download the file haarcascade_frontalface_default.xml from the official OpenCV repository.

Direct Link: haarcascade_frontalface_default.xml

Click the "Download raw file" button and save it directly into the models folder.

Configuration
Step 1: Set Up Your IP Camera
The easiest way to get started is by using your smartphone as an IP camera.

Install an App: Download an IP camera app on your phone. "IP Webcam" for Android is a great choice.

Start the Server: Launch the app and start the video server.

Get the URL: The app will display a URL on the screen, typically something like http://192.168.x.x:8080. For video, you need the "shot.jpg" or "video" endpoint.

Example URL: http://192.168.x.x:8080/shot.jpg

Step 2: Update the Configuration File
Open the src/config.py file.

Find the IP_CAMERA_URL variable and replace the placeholder with the URL from your camera app.

How to Use the System
Run all commands from the root directory of the project (face_recognition_project/).

1. Collect Face Samples
This script will capture 100 photos of a person's face from the camera feed. For the best results, train the model with images of at least 5-10 different people.

Run the command for each person you want to add:

python main.py collect

The system will then prompt you to enter the person's name.

2. Preprocess the Data
After collecting images for all individuals, process them to prepare for training. This step resizes, normalizes, and splits the data into training and testing sets.

python main.py preprocess

3. Train the Model
This command starts the training process using the preprocessed data. It will use data augmentation to improve accuracy and will save the best-performing model to the models directory.

python main.py train

After training, it will display a performance report and a confusion matrix.

4. Run Live Face Recognition
Once the model is trained, you can start the live recognition.

python main.py run

A window will open showing your camera feed with recognized faces labeled with their names and a confidence score. Press 'q' to quit.

(Optional) Check Your Setup
You can use this command to verify that the Haar Cascade file is correctly placed and readable by OpenCV.

python main.py check

