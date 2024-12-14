import os
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import load_and_preprocess_dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Path to the saved model
model_path = 'models/emotion_recognition_resnet50.h5'

# Check if the model exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Train and save the model first.")

# Load the saved model
print("Loading the trained model...")
model = load_model(model_path)
print("Model loaded successfully!")

# Load and preprocess the test data
print("Loading and preprocessing test data...")
dataset_path = os.path.abspath('datasets/FER2013/fer2013.csv')
(_, _), (_, _), (X_test, y_test) = load_and_preprocess_dataset(dataset_path)

# Convert grayscale to RGB
import tensorflow as tf
X_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_test))
print("Test data shape after RGB conversion:", X_test.shape)

# Evaluate the model on the test data
print("Evaluating the model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Generate predictions
print("Generating predictions...")
y_pred = model.predict(X_test, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification report
target_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=target_names))

# Confusion matrix
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print(conf_matrix)

# Visualize confusion matrix
plt.figure(figsize=(8, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(np.arange(len(target_names)), target_names, rotation=45)
plt.yticks(np.arange(len(target_names)), target_names)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()
