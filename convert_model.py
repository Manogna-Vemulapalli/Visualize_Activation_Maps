from tensorflow.keras.models import model_from_json, save_model

# Step 1: Load the model architecture from JSON
with open("model_a1.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

# Step 2: Load the model weights from H5 file
model.load_weights("model_weights1.h5")

# Step 3: Save the full model in Keras 3 format (.keras)
save_model(model, "emotion_model.keras")  # This will save in current directory

print("✅ Model successfully rebuilt and saved as emotion_model.keras")
