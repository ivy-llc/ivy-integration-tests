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

    TranspiledTFeat = ivy.transpile(kornia.feature.TFeat, source="torch", target=target_framework)

    x = torch.rand(16, 1, 32, 32)
    torch_out = kornia.feature.TFeat()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledTFeat()(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_SOSNet(target_framework, mode, backend_compile):
    print("kornia.feature.SOSNet")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledSOSNet = ivy.transpile(kornia.feature.SOSNet, source="torch", target=target_framework)

    x = torch.rand(8, 1, 32, 32)
    torch_out = kornia.feature.SOSNet()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledSOSNet()(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_BlobHessian(target_framework, mode, backend_compile):
    print("kornia.feature.BlobHessian")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledBlobHessian = ivy.transpile(kornia.feature.BlobHessian, source="torch", target=target_framework)

    x = torch.rand(1, 1, 32, 32)
    torch_out = kornia.feature.BlobHessian()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledBlobHessian()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_CornerGFTT(target_framework, mode, backend_compile):
    print("kornia.feature.CornerGFTT")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledCornerGFTT = ivy.transpile(kornia.feature.CornerGFTT, source="torch", target=target_framework)

    x = torch.rand(1, 1, 32, 32)
    torch_out = kornia.feature.CornerGFTT()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledCornerGFTT()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_CornerHarris(target_framework, mode, backend_compile):
    print("kornia.feature.CornerHarris")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledCornerHarris = ivy.transpile(kornia.feature.CornerHarris, source="torch", target=target_framework)

    x = torch.rand(1, 1, 32, 32)
    torch_out = kornia.feature.CornerHarris(x)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledCornerHarris(transpiled_x)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_BlobDoG(target_framework, mode, backend_compile):
    print("kornia.feature.BlobDoG")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledBlobDoG = ivy.transpile(kornia.feature.BlobDoG, source="torch", target=target_framework)

    x = torch.rand(2, 3, 3, 32, 32)
    torch_out = kornia.feature.BlobDoG()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledBlobDoG()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_BlobDoGSingle(target_framework, mode, backend_compile):
    print("kornia.feature.BlobDoGSingle")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledBlobDoGSingle = ivy.transpile(kornia.feature.BlobDoGSingle, source="torch", target=target_framework)

    x = torch.rand(1, 1, 32, 32)
    torch_out = kornia.feature.BlobDoGSingle()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledBlobDoGSingle()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_KeyNet(target_framework, mode, backend_compile):
    print("kornia.feature.KeyNet")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledKeyNet = ivy.transpile(kornia.feature.KeyNet, source="torch", target=target_framework)

    x = torch.rand(1, 1, 32, 32) * 10.
    torch_out = kornia.feature.KeyNet()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledKeyNet()(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_MultiResolutionDetector(target_framework, mode, backend_compile):
    print("kornia.feature.MultiResolutionDetector")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledMultiResolutionDetector = ivy.transpile(kornia.feature.MultiResolutionDetector, source="torch", target=target_framework)
    TranspiledKeyNet = ivy.transpile(kornia.feature.KeyNet, source="torch", target=target_framework)

    model = kornia.feature.KeyNet()
    transpiled_model = TranspiledKeyNet()

    x = torch.rand(1, 1, 32, 32) * 10.
    torch_out = kornia.feature.MultiResolutionDetector(model)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledMultiResolutionDetector(transpiled_model)(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_ScaleSpaceDetector(target_framework, mode, backend_compile):
    print("kornia.feature.ScaleSpaceDetector")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledScaleSpaceDetector = ivy.transpile(kornia.feature.ScaleSpaceDetector, source="torch", target=target_framework)

    x = torch.rand(1, 1, 32, 32) * 10.
    torch_out = kornia.feature.ScaleSpaceDetector()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledScaleSpaceDetector()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_KeyNetDetector(target_framework, mode, backend_compile):
    print("kornia.feature.KeyNetDetector")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledKeyNetDetector = ivy.transpile(kornia.feature.KeyNetDetector, source="torch", target=target_framework)

    x = torch.rand(1, 1, 32, 32)
    torch_out = kornia.feature.KeyNetDetector()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledKeyNetDetector()(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_LAFDescriptor(target_framework, mode, backend_compile):
    print("kornia.feature.LAFDescriptor")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledLAFDescriptor = ivy.transpile(kornia.feature.LAFDescriptor, source="torch", target=target_framework)

    x = torch.rand(1, 1, 64, 64)
    lafs = torch.rand(1, 2, 2, 3)
    torch_out = kornia.feature.LAFDescriptor()(x, lafs)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_lafs = _nest_torch_tensor_to_new_framework(lafs, target_framework)
    transpiled_out = TranspiledLAFDescriptor()(transpiled_x, transpiled_lafs)

    _to_numpy_and_allclose(torch_out, transpiled_out)

#TODO: figure out workaround for the segfault error. Most likely, we need to modify the
# config to reduce the overall computation overhead.
# def test_SOLD2(target_framework, mode, backend_compile):
#     print("kornia.feature.SOLD2")

    # if backend_compile or target_framework == "numpy":
    #     pytest.skip()

#     x = torch.rand(1, 1, 512, 512)
#     torch_out = kornia.feature.SOLD2(pretrained=False)(x)

#     transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
#     transpiled_out = TranspiledSOLD2(pretrained=False)(transpiled_x)

#     _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_LocalFeature(target_framework, mode, backend_compile):
    print("kornia.feature.LocalFeature")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledKeyNetDetector = ivy.transpile(
        kornia.feature.KeyNetDetector, source="torch", target=target_framework
    )
    TranspiledLAFDescriptor = ivy.transpile(
        kornia.feature.LAFDescriptor, source="torch", target=target_framework
    )
    TranspiledLocalFeature = ivy.transpile(
        kornia.feature.LocalFeature, source="torch", target=target_framework
    )

    torch_detector = kornia.feature.KeyNetDetector()
    torch_descriptor = kornia.feature.LAFDescriptor()
    transpiled_detector = TranspiledKeyNetDetector()
    transpiled_descriptor = TranspiledLAFDescriptor()

    x = torch.rand(1, 1, 128, 128)
    torch_out = kornia.feature.LocalFeature(torch_detector, torch_descriptor)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledLocalFeature(transpiled_detector, transpiled_descriptor)(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


# def test_SOLD2_detector(target_framework, mode, backend_compile):
#     print("kornia.feature.SOLD2_detector")

    # if backend_compile or target_framework == "numpy":
    #     pytest.skip()

#     TranspiledSOLD2Detector = ivy.transpile(kornia.feature.SOLD2_detector, source="torch", target=target_framework)

#     x = torch.rand(1, 1, 512, 512)
#     torch_out = kornia.feature.SOLD2_detector(pretrained=False)(x)

#     transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
#     transpiled_out = TranspiledSOLD2Detector(pretrained=False)(transpiled_x)

#     _to_numpy_and_allclose(torch_out, transpiled_out)
