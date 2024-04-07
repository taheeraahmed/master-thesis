import torch
import argparse
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.transforms import (Compose,
                                    Resize,
                                    ToTensor)
from torchmetrics.classification import MultilabelF1Score

from torchmetrics import AUROC

from models import set_model
from data import ChestXray14HFDataset

def inference():
    experiment_name = "2024-04-07-12:55:58-alexnet-focal-multi-label-e35-bs32-lr0.0005-t45:00:00"
    root_path = "/cluster/home/taheeraa/code/master-thesis/01-multi-label/output"

    experiment_path = f"{root_path}/{experiment_name}"
    output_path = f"{experiment_path}/predicted"
    checkpoint_file = f"{experiment_path}/model_checkpoints/lightning_logs/version_0/checkpoints/epoch=3-step=4100.ckpt"
    test_ids_csv = f"{root_path}/other/8_test_ids.csv"
    model_arg = "alexnet"
    num_labels = 8

    model, img_size = set_model(model_arg, num_labels)
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    adjusted_state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(adjusted_state_dict)

    test_transforms = Compose([
        Resize((img_size, img_size), antialias=True),
        ToTensor(),
        #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_df = pd.read_csv(test_ids_csv)
    test_dataset = ChestXray14HFDataset(test_df, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(test_loader.__repr__())

    # Ensure the model is in evaluation mode
    model.eval()  

    all_preds = []
    all_gt = []

    f1_score = MultilabelF1Score(
        num_labels=num_labels, 
        threshold=0.5, 
        average='macro'
    )

    f1_score_micro = MultilabelF1Score(
        num_labels=num_labels,
        threshold=0.5,
        average='micro'
    )
        
    auroc = AUROC(
        task="multilabel",
        num_labels=num_labels,
        average="macro",
    )

    # No need to track gradients for inference
    print("started inference")
    print(f"num batches: {len(test_loader)}")
    with torch.no_grad():  
        for batch in test_loader:
            images, gt = batch['pixel_values'], batch['labels']
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            f1_score.update(outputs, gt.int())
            auroc.update(probs, gt.int())

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())  
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


if __name__ == "__main__":
    inference()