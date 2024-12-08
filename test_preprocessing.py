import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

from utils.preprocess import load_and_preprocess_dataset

# Path to the FER2013 dataset
dataset_path = r'C:\Users\21693\emotion-recognition\datasets\FER2013\fer2013.csv'

try:
    # Load and preprocess the dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess_dataset(dataset_path)

    # Print shapes of the preprocessed data
    print("✅ Preprocessing Successful!")
    print("Training data shape:", X_train.shape, y_train.shape)
    print("Validation data shape:", X_val.shape, y_val.shape)
    print("Test data shape:", X_test.shape, y_test.shape)

except FileNotFoundError:
    print(f"❌ Error: Dataset file not found at '{dataset_path}'. Please ensure the file exists.")
except Exception as e:
    print(f"❌ An error occurred: {e}")
