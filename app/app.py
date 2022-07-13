import gradio as gr
import lightning as L
import numpy as np
import os
import streamlit as st
import torch
import torchvision.transforms as T
from lightning.app.components.serve import ServeGradio
from lightning.app.frontend import StreamlitFrontend
from torchvision import transforms
from components import MyServeGradioComponent
from helper import corn_label_from_logits, coral_label_from_logits, crossentr_label_from_logits, niu_label_from_logits


class PyTorchModels(MyServeGradioComponent):

    inputs = gr.inputs.Image(type="pil", label="Select an input image")  # required
    outputs = gr.outputs.Textbox(type="text")  # required
    examples = [f"./examples/{f}" for f in os.listdir('./examples') if f.endswith('.jpg')]  # required

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ready = False  # required

    def predict(self, img):
        img_tensor = self.preprocessor(img)
        img_tensor.unsqueeze_(0) # adds batch dimension

        corn_model, coral_model, niu_model, crossentr_model = self.model
        with torch.inference_mode():
            corn_pred = corn_model(img_tensor)
            coral_pred = coral_model(img_tensor)
            niu_pred = niu_model(img_tensor)
            crossentr_pred = crossentr_model(img_tensor)

        s =  "###########################"
        s += "\n###### PREDICTED AGE ######"
        s += "\n###########################"
        s += f"\n\nCORN model: {corn_label_from_logits(corn_pred).item()+18} years"
        s += f"\nCORAL model: {coral_label_from_logits(coral_pred).item()+18} years"
        s += f"\nNiu et al. model: {niu_label_from_logits(niu_pred).item()+18} years"
        s += f"\nCross entropy model: {crossentr_label_from_logits(crossentr_pred).item()+18} years"
        s += "\n\n(ResNet-34 models trained on AFAD, age range 18-30).\n"\
             "This is for demo purposes and not an accurate age predictor!"

        return s

    def build_preprocessor(self):
        preprocessing = transforms.Compose([
                transforms.CenterCrop((140, 140)),
                transforms.Resize((128, 128)),
                transforms.CenterCrop((120, 120)),
                transforms.ToTensor()])
        return preprocessing

    def build_model(self):
        DEVICE = torch.device("cpu")

        corn_model = torch.hub.load(
            "rasbt/ord-torchhub",
            model="resnet34_corn_afad",
            source='github',
            pretrained=True
        )

        coral_model = torch.hub.load(
            "rasbt/ord-torchhub",
            model="resnet34_coral_afad",
            source='github',
            pretrained=True
        )

        niu_model = torch.hub.load(
            "rasbt/ord-torchhub",
            model="resnet34_niu_afad",
            source='github',
            pretrained=True
        )

        crossentr_model = torch.hub.load(
            "rasbt/ord-torchhub",
            model="resnet34_crossentr_afad",
            source='github',
            pretrained=True
        )

        all_models = (corn_model, coral_model, niu_model, crossentr_model)
        for model in all_models:
            model.eval()
            model.to(DEVICE)

        return all_models


def your_streamlit_app(lightning_app_state):
    static_text = """
    # Ordinal Regression Model App

    This is a simple [Lightning App](https://lightning.ai) that runs various 
    ordinal regression models. As of this writing, all models are based on a ResNet-34
    base architecture and trained on the [AFAD](https://afad-dataset.github.io) dataset.

    ## Further Resources
    
    1. The research paper describing the losses and model training: 
        "Deep Neural Networks for Rank-Consistent Ordinal Regression Based On Conditional Probabilities":
        [https://arxiv.org/abs/2103.14724](https://arxiv.org/abs/2103.14724)
    2. The source code for this App: [https://github.com/rasbt/ord-torchhub/tree/main/app](https://github.com/rasbt/ord-torchhub/tree/main/app)
    3. The TorchHub repo for the pretrained models loaded into this App: [https://github.com/rasbt/ord-torchhub](https://github.com/rasbt/ord-torchhub)
    4. Tutorial material describing the main concepts behind these ordinal regression models: [https://github.com/rasbt/scipy2022-talk](https://github.com/rasbt/scipy2022-talk)
    
    If you want to learn more about Lightning Apps, checkout the official
    [lightning.ai](https://lightning.ai) website.

    If you have any questions or suggestions, please feel free to open a GitHub Issue or Discussion in one of the repositories referenced above.

    ## About the Author

    You can find out more about me at [https://sebastianraschka.com](https://sebastianraschka.com).

    """
    st.write(static_text)


class ChildFlow(L.LightningFlow):
    def configure_layout(self):
        return StreamlitFrontend(render_fn=your_streamlit_app)


class RootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()

        self.demo = PyTorchModels(cloud_compute=L.CloudCompute("cpu-medium"))
        self.about_page = ChildFlow()

    def run(self):
        self.demo.run()

    def configure_layout(self):
        tab_1 = {"name": "Ordinal Models", "content": self.demo}
        tab_2 = {
            "name": "CORN Paper",
            "content": "https://arxiv.org/pdf/2111.08851.pdf",
        }
        tab_3 = {"name": "About", "content": self.about_page}
        return tab_1, tab_2, tab_3


app = L.LightningApp(RootFlow())