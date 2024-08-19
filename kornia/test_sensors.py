from helpers import (
    _array_to_new_backend,
    _check_allclose,
    _check_shape_allclose,
    _nest_array_to_numpy,
    _nest_torch_tensor_to_new_framework,
)

import ivy
import kornia
from kornia.sensors import camera
from kornia.sensors.camera import (
    CameraModel,
    CameraModelType,
    PinholeModel,
)
import pytest
import torch


# Helpers #
# ------- #

def _to_numpy_and_allclose(torch_x, transpiled_x, tolerance=1e-3):
    orig_data = _nest_array_to_numpy(torch_x)
    transpiled_data = _nest_array_to_numpy(transpiled_x)
    _check_allclose(orig_data, transpiled_data, tolerance=tolerance) 


# Tests #
# ----- #

def test_CameraModel(target_framework, mode, backend_compile):
    print("kornia.sensors.camera.CameraModel")

    if backend_compile:
        pytest.skip()

    TranspiledImageSize = ivy.transpile(kornia.image.ImageSize, source="torch", target=target_framework)
    TranspiledCameraModelType = ivy.transpile(CameraModelType, source="torch", target=target_framework)
    TranspiledCameraModel = ivy.transpile(CameraModel, source="torch", target=target_framework)

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

    # TODO: test project() and other methods


def test_PinholeModel(target_framework, mode, backend_compile):
    print("kornia.sensors.camera.PinholeModel")

    if backend_compile:
        pytest.skip()

    TranspiledImageSize = ivy.transpile(kornia.image.ImageSize, source="torch", target=target_framework)
    TranspiledPinholeModel = ivy.transpile(PinholeModel, source="torch", target=target_framework)

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


def test_AffineTransform(target_framework, mode, backend_compile):
    print("kornia.sensors.camera.distortion_model.AffineTransform")

    if backend_compile:
        pytest.skip()

    TranspiledVector2 = ivy.transpile(kornia.geometry.vector.Vector2, source="torch", target=target_framework)
    TranspiledAffineTransform = ivy.transpile(camera.distortion_model.AffineTransform, source="torch", target=target_framework)

    params = torch.Tensor([1., 2., 3., 4.])
    points = kornia.geometry.vector.Vector2.from_coords(1., 2.)
    torch_out = camera.distortion_model.AffineTransform().distort(params, points).data

    transpiled_params = _array_to_new_backend(params, target_framework)
    transpiled_points = TranspiledVector2.from_coords(1., 2.)
    transpiled_out = TranspiledAffineTransform().distort(transpiled_params, transpiled_points).data

    _to_numpy_and_allclose(torch_out, transpiled_out)

    params = torch.Tensor([1., 2., 3., 4.])
    points = kornia.geometry.vector.Vector2.from_coords(1., 2.)
    torch_out = camera.distortion_model.AffineTransform().undistort(params, points).data

    transpiled_params = _array_to_new_backend(params, target_framework)
    transpiled_points = TranspiledVector2.from_coords(1., 2.)
    transpiled_out = TranspiledAffineTransform().undistort(transpiled_params, transpiled_points).data

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_Z1Projection(target_framework, mode, backend_compile):
    print("kornia.sensors.camera.projection_model.Z1Projection")

    if backend_compile:
        pytest.skip()

    TranspiledVector2 = ivy.transpile(kornia.geometry.vector.Vector2, source="torch", target=target_framework)
    TranspiledVector3 = ivy.transpile(kornia.geometry.vector.Vector3, source="torch", target=target_framework)
    TranspiledZ1Projection = ivy.transpile(camera.projection_model.Z1Projection, source="torch", target=target_framework)

    points = kornia.geometry.vector.Vector3.from_coords(1., 2., 3.)
    torch_out = camera.projection_model.Z1Projection().project(points).data

    transpiled_points = TranspiledVector3.from_coords(1., 2., 3.)
    transpiled_out = TranspiledZ1Projection().project(transpiled_points).data

    _to_numpy_and_allclose(torch_out, transpiled_out)

    points = kornia.geometry.vector.Vector2.from_coords(1., 2.)
    torch_out = camera.projection_model.Z1Projection().unproject(points, 3).data

    transpiled_points = TranspiledVector2.from_coords(1., 2.)
    transpiled_out = TranspiledZ1Projection().unproject(transpiled_points, 3).data

    _to_numpy_and_allclose(torch_out, transpiled_out)
