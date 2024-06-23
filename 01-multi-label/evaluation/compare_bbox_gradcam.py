import cv2
import numpy as np
import pandas as pd
import os
from evaluation.xai import xai
from evaluation.prepare_data import load_and_preprocess_images
from tqdm import tqdm

def compare_bbox_gradcam(model, model_str, dataset_path, normalization):
    gt_bboxes_df = get_dataframe_bboxes(dataset_path)
    all_ious = [] 
    
    unique_img_indices = gt_bboxes_df["Image Index"].unique()
    # Explicitly initializing tqdm with manual control over the update
    with tqdm(total=len(unique_img_indices), desc=f"Processing images") as pbar:
        for img_index in unique_img_indices:
            df_rows = gt_bboxes_df.loc[gt_bboxes_df['Image Index'] == img_index]
            label = df_rows['Finding Label'].values[0]
            gt_bboxes = get_ground_truth_bbox(df_rows)
            
            img_path = df_rows['Image File Path'].values[0]
            input_tensor = load_and_preprocess_images(img_path, normalization, one_img=True)
            gradcam_bboxes = get_gradcam_bbox(model, model_str, input_tensor, img_path, img_index)

            # size of grad_cam bbox
            gc_sizes= []
            for gradcam_bbox in gradcam_bboxes:
                x, y, w, h = gradcam_bbox
                gc_width = w
                gc_height = h
                size = gc_width * gc_height
                gc_sizes.append(size)
            avg_gc_size = sum(gc_sizes) / len(gc_sizes) if gc_sizes else 0

            # avg size of grount truth bounding box
            gt_sizes = []
            for gt_bbox in gt_bboxes:
                x, y, w, h = gt_bbox
                gt_width = w
                gt_height = h
                size = gt_width * gt_height
                gt_sizes.append(size)
            avg_gt_size = sum(gt_sizes) / len(gt_sizes) if gt_sizes else 0
                

            ious_img = calculate_ious(gradcam_bboxes, gt_bboxes)
            avg_iou_img = sum(ious_img) / len(ious_img) if ious_img else 0
            all_ious.append({"avg_iou_img": avg_iou_img, "img_index": img_index, "label": label, "avg_gc_size": avg_gc_size, "avg_gt_size": avg_gt_size, "num_gc_bbox": len(gradcam_bboxes), "num_gt_bbox": len(gt_bboxes)})
            
            # Update progress bar and set postfix after each image is processed
            pbar.set_postfix({'avg_iou': f'{avg_iou_img:.3f}', 'img_index': img_index, 'label': label})
            pbar.update(1)
        
        # Create a DataFrame from the results
    results_df = pd.DataFrame(all_ious)
    
    # Save the DataFrame to a CSV file
    results_df.to_csv(f"results/{model_str}_iou.csv", index=False)

    # Calculate the average IOU safely
    return results_df

def scale_bboxes(bboxes, orig_size=(1024, 1024), cam_size=(224, 224)):
    """
    Scales bounding boxes from Grad-CAM resolution back to original image size.

    Parameters:
    - bboxes: List of bounding boxes in the format [x, y, w, h].
    - orig_size: Tuple (width, height) of the original image size.
    - cam_size: Tuple (width, height) of the CAM image size.

    Returns:
    - scaled_bboxes: List of scaled bounding boxes in the format [x, y, w, h].
    """
    # Calculate scale factors for width and height
    scale_x = orig_size[0] / cam_size[0]
    scale_y = orig_size[1] / cam_size[1]

    # Scale each bounding box
    scaled_bboxes = []
    for x, y, w, h in bboxes:
        scaled_x = int(x * scale_x)
        scaled_y = int(y * scale_y)
        scaled_w = int(w * scale_x)
        scaled_h = int(h * scale_y)
        scaled_bboxes.append([scaled_x, scaled_y, scaled_w, scaled_h])

    return scaled_bboxes


def get_gradcam_bbox(model, model_str, input_tensor, img_path, img_id):
    """
    Generates Grad-CAM from a model and extracts bounding boxes from it.
    
    :param model: Model to generate Grad-CAM from.
    :param model_str: String representation of the model.
    :param input_tensor: Preprocessed input tensor.
    :param img_path: Path to the original image.
    :param img_id: Image ID.
    :return: List of bounding boxes from Grad-CAM.    
    """

    gradcam_bboxes = []
    cam_image = xai(model, model_str, input_tensor, img_path, img_id, get_bbox=True)

    # extract the red channel
    red_channel = cam_image[:, :, 0]

    # threshold the red channel
    thresh = cv2.threshold(red_channel, np.max(red_channel) * 0.7, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((10, 10), np.uint8)  # larger kernel
    dilation = cv2.dilate(thresh, kernel, iterations=3)

    # find contours
    contours, _ = cv2.findContours(dilation.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter contours and calculate bounding boxes
    contours = [c for c in contours if cv2.contourArea(c) > 100]  # higher area threshold
    gradcam_bboxes = [cv2.boundingRect(c) for c in contours]

    # scale bboxes to match original image size
    scaled_gradcam_bboxes = scale_bboxes(gradcam_bboxes)
    return scaled_gradcam_bboxes

def get_dataframe_bboxes(dataset_path):
    df_bbox = pd.read_csv(os.path.join(dataset_path, "BBox_List_2017.csv"))
    df_data_entry = pd.read_csv(os.path.join(dataset_path, "Data_Entry_2017.csv"))

    merged_df = pd.merge(df_bbox, df_data_entry, on='Image Index', how='inner')
    merged_df.rename(columns={
        'Bbox [x': 'x',
        'h]': 'h',
    }, inplace=True)

    merged_df['Image File Path'] = '/cluster/home/taheeraa/datasets/chestxray-14/images/' + merged_df['Image Index']
    df = merged_df[['Image Index', 'Finding Label', 'x', 'y', 'w', 'h', 'Image File Path']]
    return df

def get_ground_truth_bbox(df_rows):
    gt_bboxes = []
    for _, row in df_rows.iterrows():
        x = row['x']
        y = row['y']
        w = row['w']
        h = row['h']
        gt_bboxes.append([x, y, w, h])
    return gt_bboxes


def bbox_to_corners(bbox):
    """Converts bounding box from [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]

def calculate_iou_single(bbox1, bbox2):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    # Convert bounding boxes to [x1, y1, x2, y2]
    bbox1 = bbox_to_corners(bbox1)
    bbox2 = bbox_to_corners(bbox2)

    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate each bbox area
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Calculate union area
    union_area = bbox1_area + bbox2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area
    return iou

def calculate_ious(grad_cam_bboxes, ground_truth_bboxes):
    """Calculates IoUs between lists of bounding boxes from Grad-CAM and ground truth."""
    ious = []
    for grad_cam_bbox in grad_cam_bboxes:
        for gt_bbox in ground_truth_bboxes:
            iou = calculate_iou_single(grad_cam_bbox, gt_bbox)
            ious.append(iou)
    return ious