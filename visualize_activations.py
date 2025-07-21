import os
import matplotlib.pyplot as plt
from keras.models import load_model
from src.preprocess import preprocess_input_image
from src.layer_utils import get_intermediate_layers

def visualize_feature_maps(model_path, image_path, output_dir):
    """
    Generates and saves activation maps for intermediate layers.
    """
    # Load the trained model
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)

    # Load and preprocess the image
    print(f"Preprocessing image: {image_path}")
    preprocessed_img = preprocess_input_image(image_path)

    # Extract intermediate layers
    intermediate_layers = get_intermediate_layers(model)

    os.makedirs(output_dir, exist_ok=True)

    for layer_name, intermediate_model in intermediate_layers.items():
        print(f"Visualizing layer: {layer_name}")
        layer_activation = intermediate_model.predict(preprocessed_img)

        if layer_activation.ndim != 4:
            print(f"Skipping {layer_name}: unexpected shape {layer_activation.shape}")
            continue

        num_channels = layer_activation.shape[-1]
        num_plots = min(num_channels, 8)
        fig, axes = plt.subplots(1, num_plots, figsize=(15, 8))

        if num_plots == 1:
            axes = [axes]

        for j in range(num_plots):
            ax = axes[j]
            ax.imshow(layer_activation[0, :, :, j], cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Filter {j + 1}')

        plt.suptitle(f"Layer: {layer_name}", fontsize=14)
        save_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_{layer_name}.png")
        plt.savefig(save_path)
        print(f"Saved activation map to: {save_path}")
        plt.close()
