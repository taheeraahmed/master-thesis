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
import numpy as np

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


def grad_cam(model, model_str, input_tensor, pil_image, batch_size=32):
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
        
    # convert pil image to rgb if not
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # convert image to cv2 format
    cv_image = np.array(pil_image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    # read image and resize
    rgb_img = cv2.resize(cv_image, (224, 224))
    rgb_img = np.float32(rgb_img) / 255

    if model_str in set(["swin_simim", "swin_in22k", "vit_in1k"]):
        cam = methods["gradcam"](model=model,
                                 target_layers=target_layers,
                                 reshape_transform=reshape_transform)
    else:
        cam = methods["gradcam"](model=model,
                                 target_layers=target_layers)

    cam.batch_size = batch_size
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=None,
                        eigen_smooth=True,
                        aug_smooth=False)

    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)

    return cam_image