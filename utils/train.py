from preprocess import load_and_preprocess_dataset
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout


# Dynamically generate the dataset path
dataset_path = os.path.abspath('datasets/FER2013/fer2013.csv')

# Load Data
(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess_dataset(dataset_path)
print("Training data shape:", X_train.shape, y_train.shape)
print("Validation data shape:", X_val.shape, y_val.shape)
print("Test data shape:", X_test.shape, y_test.shape)

# Convert from Grayscale to RGB
X_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_train))
X_val = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_val))
X_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_test))
print("Training data shape after RGB conversion:", X_train.shape)


# Define the data augmentation pipeline
data_gen = ImageDataGenerator(
    rotation_range=15,        # Random rotation within 15 degrees
    width_shift_range=0.1,    # Random horizontal shift
    height_shift_range=0.1,   # Random vertical shift
    zoom_range=0.1,           # Random zoom
    horizontal_flip=True      # Random horizontal flip
)

# Apply the augmentation pipeline to the training data
train_generator = data_gen.flow(X_train, y_train, batch_size=32)

# Validate by printing a single batch
for batch_images, batch_labels in train_generator:
    print("Batch shape (images):", batch_images.shape)
    print("Batch shape (labels):", batch_labels.shape)
    break


# Load ResNet50 pre-trained model without the top classification layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Print the base model summary for confirmation
print("Base ResNet50 Model:")
base_model.summary()


