import torch
from torch.utils.data import DataLoader
import pandas as pd
from torchvision.transforms import Compose, Resize, ToTensor
from torchmetrics.classification import MultilabelF1Score
from torchmetrics import AUROC
import glob
import matplotlib.pyplot as plt
import numpy as np

from models import set_model
from data import ChestXray14HFDataset
from utils import create_directory_if_not_exists

# Function to display a single image, its true labels, and its predicted labels
def save_sample(image, true_labels, predicted_labels, output_path, i):
    """
    Display the image along with its true and predicted labels.
    """
    # Assuming image is a torch.Tensor, convert it to a numpy array and transpose it
    if torch.is_tensor(image):
        image = image.numpy()
    if image.shape[0] == 3:  # If it's a 3-channel image
        image = np.transpose(image, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    
    plt.figure(figsize=(10, 8))  # Optional: Adjust figure size
    plt.imshow(image, cmap='gray')
    plt.title("Sample CXR")
    plt.xlabel(f"True: {', '.join(true_labels)}\nPredicted: {', '.join(predicted_labels)}")
    plt.xticks([])  # Remove x-axis tick marks
    plt.yticks([])  # Remove y-axis tick mark

def inference(experiment_name, model_arg, labels):
    root_path = "/cluster/home/taheeraa/code/master-thesis/01-multi-label/output"
    experiment_path = f"{root_path}/{experiment_name}"
    output_path = f"{experiment_path}/predicted"
    create_directory_if_not_exists(output_path)
    
    # Automatically find the checkpoint file
    checkpoint_files = glob.glob(f"{experiment_path}/model_checkpoints/lightning_logs/version_0/checkpoints/*.ckpt")
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint file found in {experiment_path}/model_checkpoints/lightning_logs/version_0/checkpoints/")
    checkpoint_file = checkpoint_files[0]  # Assuming there's only one file
    
    test_ids_csv = f"{root_path}/other/8_test_ids.csv"
    num_labels = len(labels)
    model, img_size = set_model(model_arg, num_labels)
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    adjusted_state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(adjusted_state_dict)

    test_transforms = Compose([
        Resize((img_size, img_size), antialias=True),
        ToTensor(),
    ])

    test_df = pd.read_csv(test_ids_csv)
    test_dataset = ChestXray14HFDataset(test_df, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(test_loader.__repr__())

    model.eval()  # Ensure the model is in evaluation mode

    all_gt = []

    f1_score = MultilabelF1Score(num_labels=num_labels, threshold=0.5, average='macro')
    f1_score_micro = MultilabelF1Score(num_labels=num_labels, threshold=0.5, average='micro')
    auroc = AUROC(task="multilabel", num_labels=num_labels, average="macro")

    print("started inference")
    print(f"num batches: {len(test_loader)}")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images, gt = batch['pixel_values'], batch['labels']
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            predicted = probs > 0.5  # Apply threshold to get binary predictions

            gt_list = gt.squeeze().tolist()
            predicted_list = predicted.squeeze().tolist()

            if i < 10:
                save_sample(
                    images.squeeze().numpy(), 
                    [labels[j] for j in range(len(gt_list)) if gt_list[j] == 1],
                    [labels[j] for j in range(len(predicted_list)) if predicted_list[j] == 1],
                    output_path,
                    i
                )
            else: 
                continue

            f1_score.update(probs, gt.int())
            f1_score_micro.update(probs, gt.int())
            auroc.update(probs, gt.int())

            all_gt.extend(gt.cpu().numpy())  
    print("finished inference")
    
    final_f1_score = f1_score.compute()
    final_auroc = auroc.compute()
    final_f1_score_micro = f1_score_micro.compute()

    results_file_path = f"{output_path}/test_results.txt"

    with open(results_file_path, "w") as file:
        file.write(f"Final F1 Score: {final_f1_score}\n")
        file.write(f"Final AUROC: {final_auroc}\n")
        file.write(f"Final F1 Score Micro: {final_f1_score_micro}\n")

def main():
    experiment_names = [
        #"2024-04-08-12:37:18-alexnet-wmlsm-multi-label-e35-bs32-lr0.0005-t20:00:00",
        "2024-04-08-12:37:06-alexnet-mlsm-multi-label-e35-bs32-lr0.0005-t20:00:00",
        "2024-04-08-12:35:45-alexnet-bce-multi-label-e35-bs32-lr0.0005-t20:00:00",
        #"2024-04-07-22:48:38-alexnet-wbce-multi-label-e35-bs32-lr0.0005-t45:00:00"
    ]
    model_arg = "alexnet"
    labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
    
    for experiment_name in experiment_names:
        print(f"Processing {experiment_name}")
        inference(experiment_name, model_arg, labels)

if __name__ == "__main__":
    main()
