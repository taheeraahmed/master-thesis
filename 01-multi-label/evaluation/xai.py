from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
import cv2
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import os

methods = {
    "gradcam": GradCAM,
    "scorecam": ScoreCAM,
    "gradcam++": GradCAMPlusPlus,
    "ablationcam": AblationCAM,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "layercam": LayerCAM,
    "fullgrad": FullGrad
}


def xai(model, model_str, input_tensor, img_path, img_id, output_folder=None, get_bbox=False):
    if model_str == "densenet121":
        target_layers = [model.features.norm5]

    elif model_str == "swin_simim" or model_str == "swin_in22k":
        target_layers = [model.layers[-1].blocks[-1].norm2]

        def reshape_transform(tensor, height=7, width=7):
            result = tensor.reshape(tensor.size(0),
                                    height, width, tensor.size(2))
            result = result.transpose(2, 3).transpose(1, 2)
            return result

    elif model_str == "vit_in1k":
        target_layers = [model.blocks[-1].norm1]

        def reshape_transform(tensor, height=14, width=14):
            result = tensor[:, 1:, :].reshape(tensor.size(0),
                                              height, width, tensor.size(2))
            result = result.transpose(2, 3).transpose(1, 2)
            return result

    rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255

    if model_str in set(["swin_simim", "swin_in22k", "vit_in1k"]):
        cam = methods["gradcam"](model=model,
                                 target_layers=target_layers,
                                 reshape_transform=reshape_transform)
    else:
        cam = methods["gradcam"](model=model,
                                 target_layers=target_layers)

    cam.batch_size = 32
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=None,
                        eigen_smooth=True,
                        aug_smooth=False)

    grayscale_cam = grayscale_cam[0, :]
    if model_str in set(["swin_simim", "swin_in22k", "vit_in1k"]):
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    else:
        cam_image = show_cam_on_image(rgb_img, grayscale_cam)

    if get_bbox:
        return cam_image
    else:
        filename = f"{output_folder}/{img_id}_grad_cam.png"
        cv2.imwrite(
            filename, cam_image)
        return None


def get_ground_truth_labels(df, img_path, img_index, img_id, output_folder):
    col_img_index_df = df[df['Image Index'] == img_index]

    # Define specific colors
    colors_list = ['#BAA4C7', '#FFC8DD', '#FFAFCC', '#BDE0FE']
    unique_labels = col_img_index_df['Finding Label'].unique()
    if len(unique_labels) > len(colors_list):
        print("Warning: There are more unique labels than provided colors. Some labels will have the same color.")
    label_color_map = {label: colors_list[i % len(
        colors_list)] for i, label in enumerate(unique_labels)}

    # Open the image
    img = Image.open(img_path)

    # Create a figure with two subplots side by side
    fig, axs = plt.subplots(1, 1, figsize=(20, 10))

    # First subplot: Image with bounding boxes
    ax1 = axs
    ax1.imshow(img, cmap='gray')

    # Draw each bounding box with its corresponding label
    for idx, row in col_img_index_df.iterrows():
        x, y, w, h = row['x'], row['y'], row['w'], row['h']
        label = row['Finding Label']
        color = label_color_map[label]
        rect = patches.Rectangle((x, y), w, h, linewidth=4,
                                 edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
        ax1.text(x, y, label, verticalalignment='top', color='black', fontsize=12, weight='bold',
                 bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.2'))

    ax1.axis('off')
    filename = f"{output_folder}/{img_id}_bboxes.png"
    print(f"Saving {filename}")
    plt.savefig(filename)
