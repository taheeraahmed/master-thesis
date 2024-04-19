import requests
import re
from typing import List

import gradio as gr
import numpy as np
import onnxruntime as ort
from torch.nn.functional import sigmoid
import torch
from torchvision import transforms
from PIL import Image

def load_label_data():
    labels = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Effusion",
        "Emphysema",
        "Fibrosis",
        "Hernia",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pleural Thickening",
        "Pneumonia",
        "Pneumothorax"
    ]
    return labels

class WebUI:
    def __init__(self):
        super().__init__()
        self.labels = load_label_data()
        self.num_labels = len(self.labels)
        self.image_path = "/cluster/home/taheeraa/code/master-thesis/04-gradio/example_images"
        
        self.checkpoint_path = "/cluster/home/taheeraa/code/master-thesis/01-multi-label/output/change scheduler to cosine and contrast aug/2024-04-16-20:58:27-densenet121-focal-9-multi-label-e35-bs64-lr0.0005-contrast-new-scheduler/model_checkpoints/test-model.onnx"
        self.model = self.load_model()

    def load_model(self):
        session = ort.InferenceSession(self.checkpoint_path)
        return session
    
    def preprocess_image(self, image: Image):
        img_size = 224

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_pipeline = transforms.Compose([
            transforms.Resize(
                (256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ColorJitter(contrast=(0.9, 1.1)),
            transforms.ToTensor(),
            normalize,
        ])

        transformed_image = transform_pipeline(image)
        return transformed_image.numpy()
    
    def run_model(self, image):
        ort_inputs = {self.model.get_inputs()[0].name: image.numpy()}
        ort_outs = self.model.run(None, ort_inputs)
        return torch.tensor(ort_outs[0])

    def do_inference(self, image):
        # add transforms to image
        preprocessed_image = self.preprocess_image(image)
        # transform to tensor
        preprocessed_tensor = torch.tensor(
            preprocessed_image, dtype=torch.float32) 
        # run model and create ouputs
        outputs = self.run_model(preprocessed_tensor.unsqueeze(0))

        # get probs
        probabilities = sigmoid(outputs).squeeze(0)
        # get top labels
        top_probs, top_idxs = torch.topk(probabilities, self.num_labels, largest=True)
        result_labels = [(self.labels[idx], prob.item())
                         for idx, prob in zip(top_idxs, top_probs)]
        # Convert list of tuples to dictionary
        results_dict = {label: prob for label, prob in result_labels}
        return results_dict
            
    def explain_pred(self, image):
        raise NotImplementedError("Saliency maps are not yet supported")

    def run(self):
        img_filenames = ['00000003_000.png', '00000013_004.png', '00000013_018.png', '00000013_019.png', '00000013_021.png', '00000013_035.png', '00000032_015.png']
        examples = [
            f"{self.image_path}/{filename}" for filename in img_filenames]
        
        with gr.Blocks() as demo:
            with gr.Row():
                # Accept PIL Image directly
                image = gr.Image(type="pil", height=512)
                labels = gr.Label(num_top_classes=self.num_labels)


                with gr.Column(scale=0.2, min_width=150):
                    run_btn = gr.Button(
                        "Run analysis", variant="primary", elem_id="run-button")

                    run_btn.click(
                        fn=self.do_inference,
                        inputs=image,
                        outputs=labels,
                    )

                    gr.Examples(
                        examples=examples,
                        inputs=image,
                        outputs=labels,
                        fn=self.do_inference,
                        cache_examples=True,
                    )

                
        demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)


def main():
    ui = WebUI()
    ui.run()


if __name__ == "__main__":
    main()