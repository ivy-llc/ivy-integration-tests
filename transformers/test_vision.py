from datasets import load_dataset
import ivy
import numpy as np
import torch
from transformers import (
    AutoImageProcessor,
    Swin2SRModel,
)

from helpers import _backend_compile, _target_to_simplified


def test_Swin2SR(target_framework, mode, backend_compile):
    print("Swin2SRModel")

    dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
    image = dataset["test"]["image"][0]
    image_processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")

    torch_model = Swin2SRModel.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
    torch_inputs = image_processor(image, return_tensors="pt")
    with torch.no_grad():
        torch_outputs = torch_model(**torch_inputs)
    torch_last_hidden_states = torch_outputs.last_hidden_state

    TFSwin2SRModel = ivy.source_to_source(Swin2SRModel, source="torch", target=target_framework)
    translated_model = TFSwin2SRModel.from_pretrained("caidas/swin2SR-classical-sr-x2-64")

    if backend_compile:
        translated_model = _backend_compile(translated_model, target_framework)

    transpiled_inputs = image_processor(image, return_tensors=_target_to_simplified(target_framework))
    transpiled_outputs = translated_model(training=False, **transpiled_inputs)
    transpiled_last_hidden_states = transpiled_outputs.last_hidden_state

    assert np.allclose(torch_last_hidden_states.numpy(), transpiled_last_hidden_states.numpy(), atol=1e-3)
