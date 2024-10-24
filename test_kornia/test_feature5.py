from helpers import (
    _nest_torch_tensor_to_new_framework,
    _test_function,
    _to_numpy_and_allclose,
    _to_numpy_and_shape_allclose,
)

import ivy
import kornia
import pytest
import torch


# Tests #
# ----- #

def test_LAFOrienter(target_framework, mode, backend_compile):
    print("kornia.feature.LAFOrienter")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    laf = torch.rand(1, 2, 2, 3)
    img = torch.rand(1, 1, 32, 32)
    transpiled_laf = _nest_torch_tensor_to_new_framework(laf, target_framework)
    transpiled_img = _nest_torch_tensor_to_new_framework(img, target_framework)

    model = kornia.feature.LAFOrienter()
    torch_out = model(laf, img)

    transpiled_model = transpiled_kornia.feature.LAFOrienter()
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_laf, transpiled_img)
    
    ivy.sync_models(model, transpiled_model)

    transpiled_out = transpiled_model(transpiled_laf, transpiled_img)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_PatchDominantGradientOrientation(target_framework, mode, backend_compile):
    print("kornia.feature.PatchDominantGradientOrientation")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    patch = torch.rand(10, 1, 32, 32)
    transpiled_patch = _nest_torch_tensor_to_new_framework(patch, target_framework)

    model = kornia.feature.PatchDominantGradientOrientation()
    torch_out = model(patch)

    transpiled_model = transpiled_kornia.feature.PatchDominantGradientOrientation()
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_patch)
    
    ivy.sync_models(model, transpiled_model)

    transpiled_out = transpiled_model(transpiled_patch)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_OriNet(target_framework, mode, backend_compile):
    print("kornia.feature.OriNet")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    patch = torch.rand(16, 1, 32, 32)
    transpiled_patch = _nest_torch_tensor_to_new_framework(patch, target_framework)

    model = kornia.feature.OriNet()
    torch_out = model(patch)

    transpiled_model = transpiled_kornia.feature.OriNet()
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_patch)
    
    ivy.sync_models(model, transpiled_model)

    transpiled_out = transpiled_model(transpiled_patch)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_LAFAffNetShapeEstimator(target_framework, mode, backend_compile):
    print("kornia.feature.LAFAffNetShapeEstimator")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    laf = torch.rand(10, 2, 2, 3)
    img = torch.rand(10, 1, 32, 32)
    torch_out = kornia.feature.LAFAffNetShapeEstimator()(laf, img)

    transpiled_laf = _nest_torch_tensor_to_new_framework(laf, target_framework)
    transpiled_img = _nest_torch_tensor_to_new_framework(img, target_framework)
    transpiled_out = transpiled_kornia.feature.LAFAffNetShapeEstimator()(transpiled_laf, transpiled_img)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_FilterResponseNorm2d(target_framework, mode, backend_compile):
    print("kornia.feature.FilterResponseNorm2d")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    x = torch.rand(1, 3, 8, 8)
    torch_out = kornia.feature.FilterResponseNorm2d(3)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = transpiled_kornia.feature.FilterResponseNorm2d(3)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_TLU(target_framework, mode, backend_compile):
    print("kornia.feature.TLU")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    x = torch.rand(1, 3, 8, 8)
    torch_out = kornia.feature.TLU(3)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = transpiled_kornia.feature.TLU(3)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_DeFMO(target_framework, mode, backend_compile):
    print("kornia.feature.DeFMO")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    x = torch.rand(2, 6, 240, 320)
    torch_out = kornia.feature.DeFMO()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = transpiled_kornia.feature.DeFMO()(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_DeDoDe(target_framework, mode, backend_compile):
    print("kornia.feature.DeDoDe")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    x = torch.rand(1, 3, 256, 256)
    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)

    model = kornia.feature.DeDoDe(amp_dtype=torch.float32)
    torch_out = model(x)

    ivy.set_backend(target_framework)
    transpiled_model = transpiled_kornia.feature.DeDoDe(amp_dtype=ivy.as_native_dtype("float32"))
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_x)
    
    ivy.sync_models(model, transpiled_model)

    transpiled_out = transpiled_model(transpiled_x)
    
    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_DISK(target_framework, mode, backend_compile):
    print("kornia.feature.DISK")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    x = torch.rand(1, 3, 32, 32)
    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)

    class TorchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(32, 32)

        def forward(self, x):
            return self.fc(x)

    torch_unet = TorchModel()
    transpiled_unet = ivy.transpile(TorchModel, target=target_framework)()

    model = kornia.feature.DISK(desc_dim=2, unet=torch_unet)
    torch_out = model(x)

    transpiled_model = transpiled_kornia.feature.DISK(desc_dim=2, unet=transpiled_unet)
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_x)

    ivy.sync_models(model, transpiled_model)

    transpiled_out = transpiled_model(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out[0].keypoints, transpiled_out[0].keypoints)
    _to_numpy_and_shape_allclose(torch_out[0].descriptors, transpiled_out[0].descriptors)
    _to_numpy_and_shape_allclose(torch_out[0].detection_scores, transpiled_out[0].detection_scores)


def test_DISKFeatures(target_framework, mode, backend_compile):
    print("kornia.feature.DISKFeatures")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)
    keypoints = torch.rand(100, 2)
    descriptors = torch.rand(100, 128)
    detection_scores = torch.rand(100)
    torch_out = kornia.feature.DISKFeatures(keypoints, descriptors, detection_scores)

    transpiled_keypoints = _nest_torch_tensor_to_new_framework(keypoints, target_framework)
    transpiled_descriptors = _nest_torch_tensor_to_new_framework(descriptors, target_framework)
    transpiled_detection_scores = _nest_torch_tensor_to_new_framework(detection_scores, target_framework)
    transpiled_out = transpiled_kornia.feature.DISKFeatures(transpiled_keypoints, transpiled_descriptors, transpiled_detection_scores)

    _to_numpy_and_shape_allclose(torch_out.keypoints, transpiled_out.keypoints)
    _to_numpy_and_shape_allclose(torch_out.descriptors, transpiled_out.descriptors)
    _to_numpy_and_shape_allclose(torch_out.detection_scores, transpiled_out.detection_scores)
