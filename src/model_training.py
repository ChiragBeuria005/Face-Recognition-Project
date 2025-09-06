import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from . import config

def build_model(num_classes):
    """
    Builds a more robust CNN model for face recognition.
    """
    model = Sequential([
        Input(shape=config.MODEL_INPUT_SHAPE),
        Conv2D(64, (3, 3), activation="relu", padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation="relu", padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation="relu", padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        
        Conv2D(256, (3, 3), activation="relu", padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        
        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=['accuracy']
    )
    return model

def plot_training_history(history):
    """
    Plots the training and validation accuracy and loss.
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test, encoder):
    """
    Evaluates the model and prints a classification report and confusion matrix.
    """
    print("\n--- Model Evaluation ---")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Classification Report
    print("\nClassification Report:")
    class_names = encoder.classes_
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def train_model():
    """
    Loads preprocessed data, trains the model using data augmentation, 
    evaluates it, and saves the best version.
    """
    print("Loading preprocessed data for training...")
    try:
        X_train = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'X_train.npy'))
        X_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'X_test.npy'))
        y_train = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'y_train.npy'))
        y_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'y_test.npy'))
        with open(config.ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)
    except FileNotFoundError:
        print("Processed data not found. Please run 'python main.py preprocess' first.")
        return

    num_classes = y_train.shape[1]
    
    model = build_model(num_classes)
    model.summary()

    # --- New: Data Augmentation ---
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_generator = datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE)

    # Callbacks for optimizing training
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(config.MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    
    print("\nStarting model training with data augmentation...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )

    print("\nTraining complete. Best model saved to:", config.MODEL_PATH)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the final model on the test set
    evaluate_model(model, X_test, y_test, encoder)
