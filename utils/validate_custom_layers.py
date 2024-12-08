from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np 

# Load the ResNet50 base model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of ResNet50
x = Flatten()(base_model.output)               # Flatten the feature maps
x = Dense(128, activation='relu')(x)           # Dense layer with 128 neurons
x = Dropout(0.5)(x)                            # Dropout for regularization
output = Dense(7, activation='softmax')(x)     # Output layer for 7 emotion classes

# Combine base model and custom layers
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary for validation
print("Model with Custom Layers:")
model.summary()


#Test on Dummy Data 
# Generate dummy data
dummy_input = np.random.rand(1, 48, 48, 3)  # A single 48x48 RGB image
dummy_output = model.predict(dummy_input)

print("Dummy Input Shape:", dummy_input.shape)
print("Dummy Output (Predicted Probabilities):", dummy_output)
print("Output Shape:", dummy_output.shape)