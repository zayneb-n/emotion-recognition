from tensorflow.keras.applications import ResNet50

# Load the ResNet50 base model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Validate that all layers are frozen
print("Are all layers frozen?")
for layer in base_model.layers:
    print(f"{layer.name}: Trainable = {layer.trainable}")

# Validate input and output shapes
print("Input Shape:", base_model.input_shape)
print("Output Shape:", base_model.output_shape)

# Print the model summary
print("ResNet50 Model Summary:")
base_model.summary()
