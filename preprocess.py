
import cv2
import numpy as np

def preprocess_input_image(image_path):
    """
    Preprocess image for FER2013 model: grayscale, 48x48, normalized, shaped (1, 48, 48, 1).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    img = cv2.resize(img, (48, 48))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  # (48, 48, 1)
    img = np.expand_dims(img, axis=0)   # (1, 48, 48, 1)
    
    return img
