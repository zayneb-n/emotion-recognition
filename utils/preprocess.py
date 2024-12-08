import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_dataset(filepath):
    """
    Load and preprocess the FER2013 dataset.

    Args:
        filepath (str): Path to the FER2013 CSV file.

    Returns:
        tuple: Preprocessed training, validation, and test data:
               (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    # Load the dataset
    data = pd.read_csv(filepath)

    # Split the dataset into training, validation, and test sets based on the usage column 
    train_data = data[data['Usage'] == 'Training']
    val_data = data[data['Usage'] == 'PublicTest']
    test_data = data[data['Usage'] == 'PrivateTest']

    # Function to convert pixel strings into normalized arrays
    def preprocess_pixels(pixels):
        # Convert space-separated pixel values into a numpy array
        pixels = np.array([int(pixel) for pixel in pixels.split()], dtype='float32')
        # Reshape to 48x48 and normalize pixel values
        return pixels.reshape(48, 48, 1) / 255.0

    # Preprocess the training data
    X_train = np.array([preprocess_pixels(pixels) for pixels in train_data['pixels']])
    y_train = to_categorical(train_data['emotion'], num_classes=7)

    # Preprocess the validation data
    X_val = np.array([preprocess_pixels(pixels) for pixels in val_data['pixels']])
    y_val = to_categorical(val_data['emotion'], num_classes=7)

    # Preprocess the test data
    X_test = np.array([preprocess_pixels(pixels) for pixels in test_data['pixels']])
    y_test = to_categorical(test_data['emotion'], num_classes=7)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
