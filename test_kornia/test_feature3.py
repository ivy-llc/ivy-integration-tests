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

def test_TFeat(target_framework, mode, backend_compile):
    print("kornia.feature.TFeat")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    x = torch.rand(16, 1, 32, 32)
    torch_out = kornia.feature.TFeat()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = transpiled_kornia.feature.TFeat()(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_SOSNet(target_framework, mode, backend_compile):
    print("kornia.feature.SOSNet")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    x = torch.rand(8, 1, 32, 32)
    torch_out = kornia.feature.SOSNet()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = transpiled_kornia.feature.SOSNet()(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_BlobHessian(target_framework, mode, backend_compile):
    print("kornia.feature.BlobHessian")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    x = torch.rand(1, 1, 32, 32)
    torch_out = kornia.feature.BlobHessian()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = transpiled_kornia.feature.BlobHessian()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_CornerGFTT(target_framework, mode, backend_compile):
    print("kornia.feature.CornerGFTT")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    x = torch.rand(1, 1, 32, 32)
    torch_out = kornia.feature.CornerGFTT()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = transpiled_kornia.feature.CornerGFTT()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_CornerHarris(target_framework, mode, backend_compile):
    print("kornia.feature.CornerHarris")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    x = torch.rand(1, 1, 32, 32)
    torch_out = kornia.feature.CornerHarris(x)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = transpiled_kornia.feature.CornerHarris(transpiled_x)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_BlobDoG(target_framework, mode, backend_compile):
    print("kornia.feature.BlobDoG")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    x = torch.rand(2, 3, 3, 32, 32)
    torch_out = kornia.feature.BlobDoG()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = transpiled_kornia.feature.BlobDoG()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_BlobDoGSingle(target_framework, mode, backend_compile):
    print("kornia.feature.BlobDoGSingle")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    x = torch.rand(1, 1, 32, 32)
    torch_out = kornia.feature.BlobDoGSingle()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = transpiled_kornia.feature.BlobDoGSingle()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_KeyNet(target_framework, mode, backend_compile):
    print("kornia.feature.KeyNet")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    x = torch.rand(1, 1, 32, 32) * 10.
    torch_out = kornia.feature.KeyNet()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = transpiled_kornia.feature.KeyNet()(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_MultiResolutionDetector(target_framework, mode, backend_compile):
    print("kornia.feature.MultiResolutionDetector")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    keynet_model = kornia.feature.KeyNet()
    transpiled_keynet_model = transpiled_kornia.feature.KeyNet()

    x = torch.rand(1, 1, 32, 32) * 10.
    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)

    model = kornia.feature.MultiResolutionDetector(keynet_model)
    torch_out = model(x)

    transpiled_model = transpiled_kornia.feature.MultiResolutionDetector(transpiled_keynet_model)
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_x)

    ivy.sync_models(model, transpiled_model)

    transpiled_out = transpiled_model(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_ScaleSpaceDetector(target_framework, mode, backend_compile):
    print("kornia.feature.ScaleSpaceDetector")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    x = torch.rand(1, 1, 32, 32) * 10.
    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)

    model = kornia.feature.ScaleSpaceDetector()
    torch_out = model(x)

    transpiled_model = transpiled_kornia.feature.ScaleSpaceDetector()
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_x)
    
    ivy.sync_models(model, transpiled_model)

    transpiled_out = transpiled_model(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_KeyNetDetector(target_framework, mode, backend_compile):
    print("kornia.feature.KeyNetDetector")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    x = torch.rand(1, 1, 32, 32)
    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)

    model = kornia.feature.KeyNetDetector()
    torch_out = model(x)

    transpiled_model = transpiled_kornia.feature.KeyNetDetector()
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_x)
    
    ivy.sync_models(model, transpiled_model)

    transpiled_out = transpiled_model(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_LAFDescriptor(target_framework, mode, backend_compile):
    print("kornia.feature.LAFDescriptor")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    x = torch.rand(1, 1, 64, 64)
    lafs = torch.rand(1, 2, 2, 3)
    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_lafs = _nest_torch_tensor_to_new_framework(lafs, target_framework)

    model = kornia.feature.LAFDescriptor()
    torch_out = model(x, lafs)

    transpiled_model = transpiled_kornia.feature.LAFDescriptor()
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_x, transpiled_lafs)
    
    ivy.sync_models(model, transpiled_model)

    transpiled_out = transpiled_model(transpiled_x, transpiled_lafs)

    _to_numpy_and_allclose(torch_out, transpiled_out)

#TODO: figure out workaround for the segfault error. Most likely, we need to modify the
# config to reduce the overall computation overhead.
# def test_SOLD2(target_framework, mode, backend_compile):
#     print("kornia.feature.SOLD2")

#     if backend_compile or target_framework == "numpy":
#          pytest.skip()
#     TranspiledSOLD2 = ivy.transpile(kornia.feature.SOLD2, source="torch", target=target_framework)
#     x = torch.rand(1, 1, 512, 512)
#     torch_out = kornia.feature.SOLD2(pretrained=False)(x)

#     transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
#     transpiled_out = TranspiledSOLD2(pretrained=False)(transpiled_x)

#     _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_LocalFeature(target_framework, mode, backend_compile):
    print("kornia.feature.LocalFeature")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    torch_detector = kornia.feature.KeyNetDetector()
    torch_descriptor = kornia.feature.LAFDescriptor()
    transpiled_detector = transpiled_kornia.feature.KeyNetDetector()
    transpiled_descriptor = transpiled_kornia.feature.LAFDescriptor()

    x = torch.rand(1, 1, 128, 128)
    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)

    model = kornia.feature.LocalFeature(torch_detector, torch_descriptor)
    torch_out = model(x)

    transpiled_model = transpiled_kornia.feature.LocalFeature(transpiled_detector, transpiled_descriptor)
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_x)
    
    ivy.sync_models(model, transpiled_model)

    transpiled_out = transpiled_model(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


# def test_SOLD2_detector(target_framework, mode, backend_compile):
#     print("kornia.feature.SOLD2_detector")

#     if backend_compile or target_framework == "numpy":
#         pytest.skip()

#     TranspiledSOLD2Detector = ivy.transpile(kornia.feature.SOLD2_detector, source="torch", target=target_framework)

#     x = torch.rand(1, 1, 512, 512)
#     torch_out = kornia.feature.SOLD2_detector(pretrained=False)(x)

#     transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
#     transpiled_out = TranspiledSOLD2Detector(pretrained=False)(transpiled_x)

#     _to_numpy_and_allclose(torch_out, transpiled_out)
