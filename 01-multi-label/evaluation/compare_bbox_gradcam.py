from evaluation.xai import xai
from evaluation.prepare_data import load_and_preprocess_images


def compare_bbox_gradcam(model, model_str, img_path, df, normalization):

    # Read the dataframe and get the image_id
    ious = []
    for img_id in df["image_id"]:
        input_tensor = load_and_preprocess_images(img_path, normalization)
        gradcam_bboxes = get_gradcam_bbox(model, model_str, input_tensor, img_path, img_id)
        gt_bboxes = get_ground_truth_bbox(df, img_id)
        iou = calculate_iou(gradcam_bboxes, gt_bboxes)
        ious.append(iou)
    # Calculate the average IOU
    avg_iou = sum(ious) / len(ious)
    return avg_iou, ious


def get_gradcam_bbox(model, model_str, input_tensor, img_path, img_id):
    gradcam_bbox = []

    cam_image = xai(model, model_str, input_tensor, img_path, img_id, get_bbox=True)

    pass 

def get_ground_truth_bbox(df, img_id):
    pass

def calculate_iou(grad_cam_bboxes, ground_truth_bboxes):
    pass