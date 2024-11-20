from datasets import load_dataset
import ivy
import os
import numpy as np
import torch
from transformers import (
    AlbertConfig,
    AlbertModel,
    AutoImageProcessor,
    AutoTokenizer,
    Swin2SRConfig,
    Swin2SRModel,
)

from helpers import _backend_compile, _target_to_simplified


# Temporary Helpers #
# ----------------- #

from packaging.version import parse
import tensorflow as tf
import keras

if parse(keras.__version__).major > 2:
    KerasVariable = keras.src.backend.Variable
else:
    KerasVariable = tf.Variable


def _retrive_layer(model, key):
    if len(key.split(".")) == 1:
        return model, key

    module_path, weight_name = key.rsplit(".", 1)

    layer = model
    for attr in module_path.split("."):
        layer = getattr(layer, attr)

    return layer, weight_name


def _maybe_update_keras_layer_weights(layer, weight_name, new_weight):
    if hasattr(layer, weight_name):
        weight_var = getattr(layer, weight_name)
        if isinstance(weight_var, tf.Variable):
            weight_var.assign(tf.Variable(new_weight, dtype=weight_var.dtype))
        elif isinstance(weight_var, KerasVariable):
            weight_var.assign(
                KerasVariable(new_weight, dtype=weight_var.dtype, name=weight_var.name)
            )
        else:
            setattr(
                layer,
                weight_name,
                tf.convert_to_tensor(new_weight, dtype=weight_var.dtype),
            )
    else:
        raise AttributeError(
            f"Layer '{layer}' does not have a weight named '{weight_name}'"
        )


def _sync_models_torch(model1, model2):
    has_keras_layers = os.environ.get("USE_NATIVE_KERAS_LAYERS", None) == "true"
    transpose_weights = (
        has_keras_layers
        or os.environ.get("APPLY_TRANSPOSE_OPTIMIZATION", None) == "true"
    )

    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    buffers1 = dict(model1.named_buffers())
    buffers2 = dict(model2.named_buffers())
    key_mapping = {}
    for k in params2.keys():
        key_mapping[k.replace("pt_", "")] = k

    for k in buffers2.keys():
        key_mapping[k.replace("pt_", "")] = k

    params2 = {k.replace("pt_", ""): v for k, v in params2.items()}
    buffers2 = {k.replace("pt_", ""): v for k, v in buffers2.items()}

    assert params1.keys() == params2.keys()
    assert buffers1.keys() == buffers2.keys()

    with torch.no_grad():
        for name in params1:
            layer, weight_name = _retrive_layer(model2, key_mapping[name])

            params1_np = params1[name].cpu().detach().numpy()
            if (
                transpose_weights
                and "DepthwiseConv" in layer.__class__.__name__
                and len(params1_np.shape) == 4
            ):
                params1_np = np.transpose(params1_np, (2, 3, 0, 1))
            elif (
                transpose_weights
                and "Conv" in layer.__class__.__name__
                and len(params1_np.shape) == 4
            ):
                params1_np = np.transpose(params1_np, (2, 3, 1, 0))
            elif (
                "Dense" in layer.__class__.__name__
                and len(params1_np.shape) == 2
                and layer.built
            ):
                params1_np = np.transpose(params1_np, (1, 0))

            if layer.__class__.__name__.startswith("Keras"):
                _maybe_update_keras_layer_weights(
                    layer=layer, weight_name=weight_name, new_weight=params1_np
                )
                params2[name] = getattr(layer, weight_name)
                continue

            params2[name].assign(tf.Variable(params1_np, dtype=params2[name].dtype))

        for name in buffers1:
            layer, weight_name = _retrive_layer(model2, key_mapping[name])

            buffers1_np = buffers1[name].cpu().detach().numpy()
            if (
                transpose_weights
                and "DepthwiseConv" in layer.__class__.__name__
                and len(params1_np.shape) == 4
            ):
                params1_np = np.transpose(params1_np, (2, 3, 0, 1))
            elif (
                transpose_weights
                and "Conv" in layer.__class__.__name__
                and len(params1_np.shape) == 4
            ):
                buffers1_np = np.transpose(buffers1_np, (2, 3, 1, 0))
            elif (
                "Dense" in layer.__class__.__name__
                and len(params1_np.shape) == 2
                and layer.built
            ):
                buffers1_np = np.transpose(buffers1_np, (1, 0))

            if layer.__class__.__name__.startswith("Keras"):
                _maybe_update_keras_layer_weights(
                    layer=layer, weight_name=weight_name, new_weight=buffers1_np
                )
                buffers2[name] = getattr(layer, weight_name)
                continue

            if isinstance(buffers2[name], tf.Variable):
                buffers2[name].assign(
                    tf.Variable(buffers1_np, dtype=buffers2[name].dtype)
                )
            else:
                buffers2[name] = tf.convert_to_tensor(
                    buffers1_np, dtype=buffers2[name].dtype
                )

    for name in params1:
        layer, weight_name = _retrive_layer(model2, key_mapping[name])

        params1_np = params1[name].cpu().detach().numpy()
        params2_np = params2[name].numpy()
        if (
            transpose_weights
            and "DepthwiseConv" in layer.__class__.__name__
            and len(params2_np.shape) == 4
        ):
            params2_np = np.transpose(params2_np, (2, 3, 0, 1))
        elif (
            transpose_weights
            and "Conv" in layer.__class__.__name__
            and len(params2_np.shape) == 4
        ):
            params2_np = np.transpose(params2_np, (3, 2, 0, 1))
        elif (
            "Dense" in layer.__class__.__name__
            and len(params1_np.shape) == 2
            and layer.built
        ):
            params2_np = np.transpose(params2_np, (1, 0))

        assert np.allclose(
            params1_np, params2_np
        ), f"Mismatch found in parameters: {name}"

    for name in buffers1:
        layer, weight_name = _retrive_layer(model2, key_mapping[name])

        buffers1_np = buffers1[name].cpu().detach().numpy()
        buffers2_np = buffers2[name].numpy()

        if (
            transpose_weights
            and "DepthwiseConv" in layer.__class__.__name__
            and len(params2_np.shape) == 4
        ):
            params2_np = np.transpose(params2_np, (2, 3, 0, 1))
        elif (
            transpose_weights
            and "Conv" in layer.__class__.__name__
            and len(params2_np.shape) == 4
        ):
            buffers2_np = np.transpose(buffers2_np, (3, 2, 0, 1))
        elif (
            "Dense" in layer.__class__.__name__
            and len(params1_np.shape) == 2
            and layer.built
        ):
            buffers2_np = np.transpose(buffers2_np, (1, 0))

        assert np.allclose(
            buffers1_np, buffers2_np
        ), f"Mismatch found in buffers: {name}"


def _compute_module_dict_tf(model, prefix=""):
    _module_dict = dict()
    for key, value in model.__dict__.items():
        if isinstance(value, (tf.keras.Model, tf.keras.layers.Layer)):
                if not hasattr(value, 'named_parameters'):
                    _module_dict.update(_compute_module_dict_tf(value, prefix=f"{key}."))
                else:
                    _module_dict[prefix + key] = value
    return _module_dict


def _compute_module_dict_pt(model, keychains):
    _module_dict = dict()
    for keychain in keychains:
        keys = keychain.split('.')
        value = model
        for key in keys:
            value = getattr(value, key)
        _module_dict[keychain] = value
    return _module_dict


def _sync_models_HF_torch_to_tf(model_pt, model_tf):
    all_submods_tf = _compute_module_dict_tf(model_tf)
    all_submods_pt = _compute_module_dict_pt(model_pt, keychains=list(all_submods_tf.keys())) 

    for (pt_model, tf_model) in zip(all_submods_pt.values(), all_submods_tf.values()):
        pt_model.eval()
        tf_model.eval()
        _sync_models_torch(pt_model, tf_model)


# Tests #
# ----- #

def test_AlbertModel(target_framework, mode, backend_compile):
    print("transformers.nlp.AlbertModel")

    TranspiledAlbertModel = ivy.transpile(
        AlbertModel, source="torch", target=target_framework
    )
    albert_config = AlbertConfig(
        embedding_size=4,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=8,
        vocab_size=1000,
        num_hidden_layers=2,
    )

    tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
    torch_model = AlbertModel.from_pretrained("albert/albert-base-v2")
    if target_framework == "tensorflow":
        # TODO: fix the issue with from_pretrained not working due to name-mismatch b/w PT model and translated TF model
        transpiled_model = TranspiledAlbertModel.from_pretrained(
            "albert/albert-base-v2",
            from_pt=True,
            config=albert_config,
            ignore_mismatched_sizes=True,
        )
    else:
        # TODO: fix the from_pretrained issue with FlaxPretrainedModel class.
        transpiled_model = TranspiledAlbertModel(albert_config)

    os.environ["USE_NATIVE_FW_LAYERS"] = "true"
    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = "true"
    ivy.sync_models(torch_model, transpiled_model)

    torch_inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    torch_outputs = torch_model(**torch_inputs)

    transpiled_inputs = tokenizer("Hello, my dog is cute", return_tensors=_target_to_simplified(target_framework))
    transpiled_outputs = transpiled_model(**transpiled_inputs)

    assert np.allclose(
        torch_outputs.last_hidden_state.numpy(),
        ivy.to_numpy(transpiled_outputs.last_hidden_state),
        atol=1e-3,
    )


# TODO: ensure this works for other tensorflow versions, such as 2.15.1
def test_Swin2SR(target_framework, mode, backend_compile):
    # NOTE: this is in the form `integration.subsection.model`, the class submodule would just be transformers.Swin2SRModel
    print("transformers.vision.Swin2SRModel")

    ivy.set_backend(target_framework)
    dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
    image = dataset["test"]["image"][0]
    image_processor = AutoImageProcessor.from_pretrained(
        "caidas/swin2SR-classical-sr-x2-64"
    )

    # modify the config to avoid OOM issues
    swin2sr_config = Swin2SRConfig()
    swin2sr_config.embed_dim = 2
    swin2sr_config.depths = [2, 2]
    swin2sr_config.num_heads = [2, 2]

    torch_model = Swin2SRModel.from_pretrained(
        "caidas/swin2SR-classical-sr-x2-64",
        config=swin2sr_config,
        ignore_mismatched_sizes=True,
    )
    torch_inputs = image_processor(image, return_tensors="pt")
    with torch.no_grad():
        torch_outputs = torch_model(**torch_inputs)
    torch_last_hidden_states = torch_outputs.last_hidden_state

    TranslatedSwin2SRModel = ivy.transpile(
        Swin2SRModel, source="torch", target=target_framework
    )
    if target_framework == "tensorflow":
        # TODO: fix the issue with from_pretrained not working due to name-mismatch b/w PT model and translated TF model
        translated_model = TranslatedSwin2SRModel.from_pretrained(
            "caidas/swin2SR-classical-sr-x2-64",
            from_pt=True,
            config=swin2sr_config,
            ignore_mismatched_sizes=True,
        )
    else:
        # TODO: fix the from_pretrained issue with FlaxPretrainedModel class.
        translated_model = TranslatedSwin2SRModel(swin2sr_config)

    os.environ["USE_NATIVE_FW_LAYERS"] = "true"
    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = "true"
    ivy.sync_models(torch_model, translated_model)

    if backend_compile:
        translated_model = _backend_compile(translated_model, target_framework)

    transpiled_inputs = image_processor(
        image, return_tensors=_target_to_simplified(target_framework)
    )
    transpiled_outputs = translated_model(**transpiled_inputs)
    transpiled_last_hidden_states = transpiled_outputs.last_hidden_state

    assert np.allclose(
        torch_last_hidden_states.numpy(),
        ivy.to_numpy(transpiled_last_hidden_states),
        atol=1e-3,
    )
