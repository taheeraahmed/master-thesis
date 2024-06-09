import torch
import os
import sys
import logging
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.load_model import load_model
from evaluation.prepare_data import create_dataloader, get_bboxes, load_and_preprocess_images
from evaluation.run_inference import test_inference_gpu,test_inference_cpu, predict
from evaluation.xai import xai, get_ground_truth_labels
from evaluation.utils_eval import generate_latex_table, get_file_size

MODEL_DICT = {
    "densenet121": {
        "experiment_name": "05-change-classifier-head",
        "pretrained_weights": "densenet121_imagenet_1k_adamw_32_bce_aug_class/model.pth.tar"
    },
    "swin_in22k": {
        "experiment_name": "06-transformers-pre-trained",
        "pretrained_weights": "swin_base_imagenet_1k_sgd_64_bce_aug/model.pth.tar"
    },
    "vit_in1k": {
        "experiment_name": "06-transformers-pre-trained",
        "pretrained_weights": "vit_base_imagenet_1k_sgd_64_bce_True/model.pth.tar"
    },
    "swin_simim": {
        "experiment_name": "07-transformer-ssl",
        "pretrained_weights": "swin_base_simmim/model.pth.tar"
    },
}




def evaluate_models(args):
    if args.partition == "GPUQ":
        log_file = "logs/evaluation_gpu.log"
    elif args.partition == "CPUQ":
        log_file = "logs/evaluation_cpu.log"

    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s] %(asctime)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger()
    logger.info(args)

    batch_size = args.batch_size
    if args.test_augument is not None:
        test_augment = True
    num_labels = 14
    num_workers = args.num_workers
    
    data_path = "/cluster/home/taheeraa/datasets/chestxray-14"
    model_base_path = "/cluster/home/taheeraa/code/BenchmarkTransformers/models/classification/ChestXray14/"

    labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
              'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
              'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

    df = get_bboxes(data_path)
    inference_performances = []
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")  # Use CPU
        logger.warning("Using CPU")

    for model_str in MODEL_DICT.keys():
        logger.info(model_str.upper())

        pretrained_weights = model_base_path + \
            MODEL_DICT[model_str]["experiment_name"] + "/" + \
            MODEL_DICT[model_str]["pretrained_weights"]

        model, normalization = load_model(
            pretrained_weights, num_labels, model_str)

        if args.inference:
            dataloader_test = create_dataloader(
                data_path, normalization, test_augment, batch_size, num_workers=num_workers)

            if args.partition == "GPUQ":
                inference_performance = test_inference_gpu(args, model, model_str, dataloader_test, device, logger)
            elif args.parition == "CPUQ":
                inference_performance = test_inference_cpu(args, model, model_str, dataloader_test, device, logger)

            file_size = get_file_size(pretrained_weights)
            file_size_mb = file_size / (1024 ** 2)  # Convert bytes to megabytes
            logger.info(f"File size: {file_size_mb:.2f} MB")

            inference_performance.file_size = file_size_mb

            inference_performances.append(inference_performance)

        if args.xai:
            output_folder = "results/" + model_str
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                
            img_id = "00010828_039"
            logger.info(f"Processing image: {img_id}")
            img_index = f"{img_id}.png"
            img_path = os.path.join(data_path, "images", img_index)

            get_ground_truth_labels(df, img_path, img_index, img_id, output_folder)

            input_tensor = load_and_preprocess_images(img_path, normalization)

            predict(model, input_tensor, labels, threshold=0.5)
            xai(model, model_str, input_tensor, img_path, img_id, output_folder)

    latex_str = generate_latex_table(inference_performances)

    if not os.path.exists("results"):
        os.makedirs("results")

    with open("results/latex_table.txt", 'w') as file:
        file.write(latex_str)

    logger.info("Evaluation completed")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for running evaluation script")
    
    parser.add_argument('--num_workers', type=int, help="Number of workers for dataloader", required=False, default=8)
    parser.add_argument('--batch_size', type=int, help="Batch size for training", required=False, default=32)
    parser.add_argument('--partition', type=str, help="CPU or GPU?", required=True)
    parser.add_argument('--test_augument', action='store_true', help="Augment test data", required=False)
    parser.add_argument('--xai', action='store_true', help="Generate XAI", required=False, default=False)
    parser.add_argument('--inference', action='store_true', help="Run inference speed tests", required=False, default=False)

    args = parser.parse_args()
    print(args)
    evaluate_models(args)
