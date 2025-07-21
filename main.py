import os
from tensorflow.keras.models import load_model
from src.visualize_activations import visualize_feature_maps

# Path to your trained model
MODEL_PATH = "../best_emotion_model.h5"  # adjust path if needed
IMAGES_FOLDER = "images"
OUTPUT_FOLDER = "outputs/activation_maps"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Loop through all images in the images/ folder
for img_name in os.listdir(IMAGES_FOLDER):
    img_path = os.path.join(IMAGES_FOLDER, img_name)
    if img_path.lower().endswith((".jpg", ".jpeg", ".png")):
        print(f"Processing {img_name}...")
        visualize_feature_maps(MODEL_PATH, img_path, OUTPUT_FOLDER)

print("Activation map generation complete.")
