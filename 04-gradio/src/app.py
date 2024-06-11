from typing import List
from torch import topk
import gradio as gr
import torch
from PIL import Image
from utils import MODELS, LABELS
from preprocess_data import build_transform_classification
from load_model import load_model
from run_inference import run_inference
from xai import grad_cam


class WebUI:
    def __init__(self):
        super().__init__()
        self.labels = LABELS
        self.num_labels = len(LABELS)
        self.image_path = "/cluster/home/taheeraa/code/master-thesis/04-gradio/example_images"

        img_filenames = ['00010575_002', '00010828_039', '00011925_072',
                         '00018253_059', '00020482_032', '00026221_001']
        self.examples = [
            f"{self.image_path}/{filename}.png" for filename in img_filenames]

        self.bbox_examples = [
            f"{self.image_path}/{filename}_bbox.png" for filename in img_filenames]

        self.examples_dict = dict(zip(self.examples, self.bbox_examples))

        self.model_str = "densenet121"
        self.normalization = "imagenet"
        self.img_size = 224
        self.set_model(self.model_str)

        self.possible_models = MODELS.keys()
        self.device = self.set_device()

    def set_model(self, model_str) -> None:
        if model_str not in MODELS.keys():
            print("Model not found")
        if model_str is None:
            print("No model found")

        ckpt_file = MODELS[model_str]["ckpt_path"]
        self.model, self.normalization, self.img_size = load_model(
            ckpt_file, num_labels=self.num_labels, model_str=model_str)
        self.model.eval()

    def preprocess_image(self, image: Image) -> torch.Tensor:
        if image is None:
            print("No image found")
        transform_pipeline = build_transform_classification(
            normalize=self.normalization, crop_size=224, resize=256, tta=True)
        transformed_image = transform_pipeline(image)
        return transformed_image

    def set_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")  # Use GPU
        else:
            device = torch.device("cpu")  # Use CPU
        return device

    def classify_image(self, image):
        if image is None:
            raise ValueError("No image found")

        top10 = self.run_inference(image)
        return {self.labels[top10[1][0][i]]: float(top10[0][0][i]) for i in range(self.num_labels)}

    def run_inference(self, image: Image) -> List[float]:
        if image is None:
            raise ValueError("No image found")
        input_tensor = self.preprocess_image(image)
        outputs = run_inference(self.model, self.model_str,
                                input_tensor, self.device, tta=True)
        outputs = topk(outputs, k=self.num_labels)
        return outputs

    def explain_pred(self, image):
        if image is None:
            raise ValueError("No image found")
        input_tensor = self.preprocess_image(image)
        cam_image = grad_cam(self.model, self.model_str, input_tensor, image)
        return cam_image

    def run(self):
        with gr.Blocks() as demo:
            with gr.Row():
                gr.Markdown("""
                <h1 align="center">Image classification and explainability</h3>
                <p align="center">
                    This demo shows classification probabilities given four models fine-tuned for the chest X-ray14-dataset.
                    To create the saliency maps the Grad-CAM method is used.
                </p>
                <p align="center">HOW TO: (1) Choose a model, (2) choose an image from the examples and (3) click on "Run analysis".</p>
                """)
            with gr.Row():
                with gr.Group():
                    model_select = gr.Dropdown(
                        choices=self.possible_models,
                        label="Model selection",
                        value=self.model_str
                    )
                    with gr.Group():
                        image = gr.Image(type="pil", height=512, label="Input image", show_label=True)
                        gr.Examples(
                            label="Examples",
                            examples=self.examples,
                            inputs=image,
                        )
                        with gr.Column(scale=1, variant="panel"):
                            run_btn = gr.Button(
                                "Run analysis", variant="primary", elem_id="run-button")

            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    labels = gr.Label(num_top_classes=self.num_labels, label="Classification probabilities", show_label=True)
                with gr.Column(scale=1, variant="panel"):
                    saliency = gr.Image(
                        height=512, label="Saliency map given Grad-CAM", show_label=True)

            run_btn.click(
                fn=self.run_inference,
                inputs=image,
                outputs=labels,
            )
            run_btn.click(
                fn=lambda x: self.classify_image(x),
                inputs=image,
                outputs=labels,
            )
            run_btn.click(
                fn=lambda x: self.explain_pred(x),
                inputs=image,
                outputs=saliency,
            )
            run_btn.click(
                fn=lambda x: self.set_model(x),
                inputs=model_select,
            )

        demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)


def main():
    ui = WebUI()
    ui.run()


if __name__ == "__main__":
    main()
