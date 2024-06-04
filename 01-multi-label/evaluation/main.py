from evaluation.load_model import load_model
from evaluation.prepare_data import create_dataloader, get_bboxes, load_and_preprocess_images
from evaluation.run_inference import test_inference, predict
from evaluation.xai import xai
import torch
import os

model_dict = {
    "densenet121": {
        "experiment_name": "05-change-classifier-head",
        "pretrained_weights": "densenet121_imagenet_1k_adamw_32_bce_aug_class/model.pth.tar"
    },
    "swin_simim": {
        "experiment_name": "07-transformer-ssl",
        "pretrained_weights": "swin_base_simmim/model.pth.tar"
    },
    "swin_in22k": {
        "experiment_name": "06-transformers-pre-trained",
        "pretrained_weights": "swin_base_imagenet_1k_sgd_64_bce_aug/model.pth.tar"
    },
    "vit_in1k": {
        "experiment_name": "06-transformers-pre-trained",
        "pretrained_weights": "vit_base_imagenet_1k_sgd_64_bce_True/model.pth.tar"
    }
}

def main():
    batch_size = 32
    test_augment = False
    num_labels = 14
    test_diseases = None
    num_workers = 4

    data_path = "/cluster/home/taheeraa/datasets/chestxray-14"
    model_base_path = "/cluster/home/taheeraa/code/BenchmarkTransformers/models/classification/ChestXray14/"

    labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
              'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
              'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

    df = get_bboxes(data_path)

    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")  # Use CPU
        print("Using CPU")

    for model_str in model_dict.keys():
        print(model_str)
        pretrained_weights = model_base_path + \
            model_dict[model_str]["experiment_name"] + "/" + \
            model_dict[model_str]["pretrained_weights"]

        model, normalization = load_model(
            pretrained_weights, num_labels, model_str)

        dataloader_test = create_dataloader(
            data_path, normalization, test_augment, batch_size, num_workers=num_workers)

        _, _ = test_inference(model, dataloader_test, device)

        # # TODO: If needed :) 
        # img_id = "00010828_039"
        # img_index = f"{img_id}.png"
        # img_path = os.path.join(data_path, "images", img_index)
        # df_filtered = df[df['Image Index'] == img_index]

        # input_tensor = load_and_preprocess_images(img_path)

        # predict(model, input_tensor, threshold=0.5)

        # xai(model, model_str, input_tensor, img_path, img_id)


if __name__ == "__main__":
    main()
