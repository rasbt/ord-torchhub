import abc
import gradio
from lightning import LightningWork
from functools import partial
from types import ModuleType
from typing import Any, List, Optional
from lightning import LightningWork



class MyServeGradioComponent(LightningWork, abc.ABC):
    inputs: Any
    outputs: Any
    examples: Optional[List] = None
    enable_queue: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.inputs
        assert self.outputs
        self._preprocessor = None
        self._model = None

    @property
    def model(self):
        return self._model

    @property
    def preprocessor(self):
        return self._preprocessor

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        """Override with your logic to make a prediction."""

    @abc.abstractmethod
    def build_preprocessor(self) -> Any:
        """Override to instantiate and return your preprocessing pipeline.
        The model would be accessible under self.preprocessor
        """

    @abc.abstractmethod
    def build_model(self) -> Any:
        """Override to instantiate and return your model.
        The model would be accessible under self.model
        """

    def run(self, *args, **kwargs):
        if self._preprocessor is None:
            self._preprocessor = self.build_preprocessor()
        if self._model is None:
            self._model = self.build_model()
        fn = partial(self.predict, *args, **kwargs)
        fn.__name__ = self.predict.__name__

        #output_size = gradio.outputs.Textbox(label="Predicted age:")
        gradio.Interface(fn=fn, inputs=self.inputs, outputs=self.outputs, examples=self.examples).launch(
            server_name=self.host,
            server_port=self.port,
            enable_queue=self.enable_queue,
        )