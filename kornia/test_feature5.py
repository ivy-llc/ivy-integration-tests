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

    if backend_compile:
        pytest.skip()

    import os
    flag = os.environ.get("APPLY_TRANSPOSE_OPTIMIZATION")
    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = "false"

    TranspiledLAFOrienter = ivy.transpile(kornia.feature.LAFOrienter, source="torch", target=target_framework)

    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = flag

    laf = torch.rand(1, 2, 2, 3)
    img = torch.rand(1, 1, 32, 32)
    torch_out = kornia.feature.LAFOrienter()(laf, img)

    transpiled_laf = _nest_torch_tensor_to_new_framework(laf, target_framework)
    transpiled_img = _nest_torch_tensor_to_new_framework(img, target_framework)
    transpiled_out = TranspiledLAFOrienter()(transpiled_laf, transpiled_img)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_PatchDominantGradientOrientation(target_framework, mode, backend_compile):
    print("kornia.feature.PatchDominantGradientOrientation")

    if backend_compile:
        pytest.skip()

    import os
    flag = os.environ.get("APPLY_TRANSPOSE_OPTIMIZATION")
    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = "false"

    TranspiledPatchDominantGradientOrientation = ivy.transpile(kornia.feature.PatchDominantGradientOrientation, source="torch", target=target_framework)

    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = flag

    patch = torch.rand(10, 1, 32, 32)
    torch_out = kornia.feature.PatchDominantGradientOrientation()(patch)

    transpiled_patch = _nest_torch_tensor_to_new_framework(patch, target_framework)
    transpiled_out = TranspiledPatchDominantGradientOrientation()(transpiled_patch)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_OriNet(target_framework, mode, backend_compile):
    print("kornia.feature.OriNet")

    if backend_compile:
        pytest.skip()

    TranspiledOriNet = ivy.transpile(kornia.feature.OriNet, source="torch", target=target_framework)

    patch = torch.rand(16, 1, 32, 32)
    torch_out = kornia.feature.OriNet()(patch)

    transpiled_patch = _nest_torch_tensor_to_new_framework(patch, target_framework)
    transpiled_out = TranspiledOriNet()(transpiled_patch)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_LAFAffNetShapeEstimator(target_framework, mode, backend_compile):
    print("kornia.feature.LAFAffNetShapeEstimator")

    if backend_compile:
        pytest.skip()

    import os
    flag = os.environ.get("APPLY_TRANSPOSE_OPTIMIZATION")
    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = "false"

    TranspiledLAFAffNetShapeEstimator = ivy.transpile(kornia.feature.LAFAffNetShapeEstimator, source="torch", target=target_framework)

    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = flag

    laf = torch.rand(10, 2, 2, 3)
    img = torch.rand(10, 1, 32, 32)
    torch_out = kornia.feature.LAFAffNetShapeEstimator()(laf, img)

    transpiled_laf = _nest_torch_tensor_to_new_framework(laf, target_framework)
    transpiled_img = _nest_torch_tensor_to_new_framework(img, target_framework)
    transpiled_out = TranspiledLAFAffNetShapeEstimator()(transpiled_laf, transpiled_img)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_FilterResponseNorm2d(target_framework, mode, backend_compile):
    print("kornia.feature.FilterResponseNorm2d")

    if backend_compile:
        pytest.skip()

    TranspiledFilterResponseNorm2d = ivy.transpile(kornia.feature.FilterResponseNorm2d, source="torch", target=target_framework)

    x = torch.rand(1, 3, 8, 8)
    torch_out = kornia.feature.FilterResponseNorm2d(3)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledFilterResponseNorm2d(3)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_TLU(target_framework, mode, backend_compile):
    print("kornia.feature.TLU")

    if backend_compile:
        pytest.skip()

    TranspiledTLU = ivy.transpile(kornia.feature.TLU, source="torch", target=target_framework)

    x = torch.rand(1, 3, 8, 8)
    torch_out = kornia.feature.TLU(3)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledTLU(3)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_DeFMO(target_framework, mode, backend_compile):
    print("kornia.feature.DeFMO")

    if backend_compile:
        pytest.skip()

    TranspiledDeFMO = ivy.transpile(kornia.feature.DeFMO, source="torch", target=target_framework)

    x = torch.rand(2, 6, 240, 320)
    torch_out = kornia.feature.DeFMO()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledDeFMO()(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_DeDoDe(target_framework, mode, backend_compile):
    print("kornia.feature.DeDoDe")

    if backend_compile:
        pytest.skip()

    TranspiledDeDoDe = ivy.transpile(kornia.feature.DeDoDe, source="torch", target=target_framework)

    x = torch.rand(1, 3, 256, 256)
    torch_out = kornia.feature.DeDoDe(amp_dtype=torch.float32)(x)

    ivy.set_backend(target_framework)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledDeDoDe(amp_dtype=ivy.as_native_dtype("float32"))(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_DISK(target_framework, mode, backend_compile):
    print("kornia.feature.DISK")

    if backend_compile:
        pytest.skip()

    TranspiledDISK = ivy.transpile(kornia.feature.DISK, source="torch", target=target_framework)

    x = torch.rand(1, 3, 256, 256)
    torch_out = kornia.feature.DISK()(x)[0]

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledDISK()(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out.keypoints, transpiled_out.keypoints)
    _to_numpy_and_shape_allclose(torch_out.descriptors, transpiled_out.descriptors)
    _to_numpy_and_shape_allclose(torch_out.detection_scores, transpiled_out.detection_scores)


def test_DISKFeatures(target_framework, mode, backend_compile):
    print("kornia.feature.DISKFeatures")

    if backend_compile:
        pytest.skip()

    keypoints = torch.rand(100, 2)
    descriptors = torch.rand(100, 128)
    detection_scores = torch.rand(100)
    torch_out = kornia.feature.DISKFeatures(keypoints, descriptors, detection_scores)

    transpiled_keypoints = _nest_torch_tensor_to_new_framework(keypoints, target_framework)
    transpiled_descriptors = _nest_torch_tensor_to_new_framework(descriptors, target_framework)
    transpiled_detection_scores = _nest_torch_tensor_to_new_framework(detection_scores, target_framework)
    transpiled_out = kornia.feature.DISKFeatures(transpiled_keypoints, transpiled_descriptors, transpiled_detection_scores)

    _to_numpy_and_shape_allclose(torch_out.keypoints, transpiled_out.keypoints)
    _to_numpy_and_shape_allclose(torch_out.descriptors, transpiled_out.descriptors)
    _to_numpy_and_shape_allclose(torch_out.detection_scores, transpiled_out.detection_scores)
