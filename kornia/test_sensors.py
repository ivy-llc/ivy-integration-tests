from helpers import (
    _check_allclose,
    _check_shape_allclose,
    _nest_array_to_numpy,
    _nest_torch_tensor_to_new_framework,
)

import ivy
import kornia
from kornia.sensors.camera import (
    CameraModel,
    CameraModelType,
    PinholeModel,
)
import pytest
import torch


# Tests #
# ----- #

def test_CameraModel(target_framework, mode, backend_compile):
    print("kornia.sensors.camera.CameraModel")

    if backend_compile:
        pytest.skip()

    TranspiledImageSize = ivy.transpiled(kornia.image.ImageSize, source="torch", target=target_framework)
    TranspiledCameraModelType = ivy.transpiled(CameraModelType, source="torch", target=target_framework)
    TranspiledCameraModel = ivy.transpiled(CameraModel, source="torch", target=target_framework)

    torch_cam = CameraModel(
        kornia.image.ImageSize(480, 640),
        CameraModelType.ORTHOGRAPHIC,
        torch.Tensor([328., 328., 320., 240.]),
    )
    transpiled_cam = TranspiledCameraModel(
        TranspiledImageSize(480, 640),
        TranspiledCameraModelType.ORTHOGRAPHIC,
        torch.Tensor([328., 328., 320., 240.]),
    )

    orig_np = _nest_array_to_numpy(torch_cam.params)
    transpiled_np = _nest_array_to_numpy(transpiled_cam.params)
    _check_allclose(orig_np, transpiled_np)

    assert torch_cam.height == transpiled_cam.height


# def test_CameraModel_project(target_framework, mode, backend_compile):
#     print("kornia.sensors.camera.CameraModel.project")

#     if backend_compile:
#         pytest.skip()

#     pytest.skip()  # TODO: add this test and unproject


def test_PinholeModel(target_framework, mode, backend_compile):
    print("kornia.sensors.camera.PinholeModel")

    if backend_compile:
        pytest.skip()

    TranspiledImageSize = ivy.transpiled(kornia.image.ImageSize, source="torch", target=target_framework)
    TranspiledPinholeModel = ivy.transpiled(PinholeModel, source="torch", target=target_framework)

    torch_cam = PinholeModel(
        kornia.image.ImageSize(480, 640),
        torch.Tensor([328., 328., 320., 240.]),
    )
    transpiled_cam = TranspiledPinholeModel(
        TranspiledImageSize(480, 640),
        torch.Tensor([328., 328., 320., 240.]),
    )

    orig_np = _nest_array_to_numpy(torch_cam.params)
    transpiled_np = _nest_array_to_numpy(transpiled_cam.params)
    _check_allclose(orig_np, transpiled_np)

    assert torch_cam.height == transpiled_cam.height


# TODO: there are more classes to test here
