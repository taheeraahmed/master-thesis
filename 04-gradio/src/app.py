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


        self.model_str = "densenet121"
        self.normalization = None
        self.img_size = None
        self.model_name = None
        self.set_model(self.model_str)

        self.possible_models = MODELS.keys()
        self.device = self.set_device()

    def set_model(self, model_str) -> None:
        ckpt_file = MODELS[model_str]["ckpt_path"]
        self.model, self.normalization, self.img_size = load_model(
            ckpt_file, num_labels=self.num_labels, model_str=model_str)
        self.model.eval()

    def preprocess_image(self, image: Image) -> torch.Tensor:

        transform_pipeline = build_transform_classification(
            normalize=self.normalization, crop_size=224, resize=256, tta=True)
        transformed_image = transform_pipeline(image)
        return transformed_image

    def set_device(selv):
        if torch.cuda.is_available():
            device = torch.device("cuda")  # Use GPU
        else:
            device = torch.device("cpu")  # Use CPU
        return device

    def classify_image(self, image):
        top10 = self.run_inference(image)
        return {self.labels[top10[1][0][i]]: float(top10[0][0][i]) for i in range(self.num_labels)}

    def run_inference(self, image: Image) -> List[float]:
        input_tensor = self.preprocess_image(image)
        outputs = run_inference(self.model, self.model_str,
                                input_tensor, self.device, tta=True)
        outputs = topk(outputs, k=self.num_labels)
        return outputs

    def explain_pred(self, image):
        print(type(image))
        input_tensor = self.preprocess_image(image)
        cam_image = grad_cam(self.model, self.model_str, input_tensor, image)
        return cam_image

    def run(self):
        img_filenames = ['00010575_002', '00010828_039', '00011925_072',
                         '00018253_059', '00020482_032', '00026221_001']
        examples = [
            f"{self.image_path}/{filename}.png" for filename in img_filenames]

        bbox_examples = [
            f"{self.image_path}/{filename}_bbox.png" for filename in img_filenames]

        print(self.labels)

        with gr.Blocks() as demo:

            run_btn = gr.Button(
                "Run analysis", variant="primary", elem_id="run-button")
            with gr.Row():
                with gr.Column():
                    labels = gr.Label(num_top_classes=self.num_labels)
                    
                    image = gr.Image(type="pil", height=512)
                    gr.Examples(
                        examples=examples,
                        inputs=image,
                    )

                with gr.Column():
                    saliency = gr.Image(
                        height=512, label="saliency map", show_label=True)

                with gr.Column(scale=0.2, min_width=150):
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
            with gr.Row():
                # TODO: Add support for model selection
                model_select = gr.Dropdown(
                    choices=self.possible_models, label="Model")
                # TODO: Add bbox image
                bbox_image = gr.Image(type="pil", height=512)

        demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)


def main():
    ui = WebUI()
    ui.run()


if __name__ == "__main__":
    main()
