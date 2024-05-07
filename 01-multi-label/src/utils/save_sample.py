import matplotlib.pyplot as plt
import numpy as np
import torch


def save_sample(image, true_labels, predicted_labels, output_path, i):
    """
    Display the image along with its true and predicted labels.
    """
    # Assuming image is a torch.Tensor, convert it to a numpy array and transpose it
    if torch.is_tensor(image):
        image = image.numpy()
    if image.shape[0] == 3:  # If it's a 3-channel image
        # Convert from (C, H, W) to (H, W, C)
        image = np.transpose(image, (1, 2, 0))

    plt.figure(figsize=(10, 8))  # Optional: Adjust figure size
    plt.imshow(image, cmap='gray')
    plt.title("Sample CXR")
    plt.xlabel(
        f"True: {', '.join(true_labels)}\nPredicted: {', '.join(predicted_labels)}")
    plt.xticks([])  # Remove x-axis tick marks
    plt.yticks([])  # Remove y-axis tick mark
    plt.savefig(f"{output_path}/sample_{i}.png")
    print(f"Saved sample_{i}.png")
    plt.close()
