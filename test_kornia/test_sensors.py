from helpers import (
    _array_to_new_backend,
    _check_allclose,
    _nest_array_to_numpy,
    _to_numpy_and_allclose,
)

import ivy
import kornia
import kornia.sensors
import kornia.sensors.camera
import pytest
import torch


# Tests #
# ----- #

def test_CameraModel(target_framework, mode, backend_compile):
    print("kornia.sensors.camera.CameraModel")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    torch_cam = kornia.sensors.camera.CameraModel(
        kornia.image.ImageSize(480, 640),
        kornia.sensors.camera.CameraModelType.ORTHOGRAPHIC,
        torch.Tensor([328., 328., 320., 240.]),
    )
    transpiled_cam = transpiled_kornia.sensors.camera.CameraModel(
        transpiled_kornia.image.ImageSize(480, 640),
        transpiled_kornia.sensors.camera.CameraModelType.ORTHOGRAPHIC,
        _array_to_new_backend(torch.Tensor([328., 328., 320., 240.]), target_framework),
    )

    orig_np = _nest_array_to_numpy(torch_cam.params)
    transpiled_np = _nest_array_to_numpy(transpiled_cam.params)
    _check_allclose(orig_np, transpiled_np)

    assert torch_cam.height == transpiled_cam.height

    # TODO: test project() and other methods


def test_PinholeModel(target_framework, mode, backend_compile):
    print("kornia.sensors.camera.PinholeModel")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    torch_cam = kornia.sensors.camera.PinholeModel(
        kornia.image.ImageSize(480, 640),
        torch.Tensor([328., 328., 320., 240.]),
    )
    transpiled_cam = transpiled_kornia.sensors.camera.PinholeModel(
        transpiled_kornia.image.ImageSize(480, 640),
        _array_to_new_backend(torch.Tensor([328., 328., 320., 240.]), target_framework),
    )

    orig_np = _nest_array_to_numpy(torch_cam.params)
    transpiled_np = _nest_array_to_numpy(transpiled_cam.params)
    _check_allclose(orig_np, transpiled_np)

    assert torch_cam.height == transpiled_cam.height


def test_AffineTransform(target_framework, mode, backend_compile):
    print("kornia.sensors.camera.distortion_model.AffineTransform")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    params = torch.Tensor([1., 2., 3., 4.])
    points = kornia.geometry.vector.Vector2.from_coords(1., 2.)
    torch_out = kornia.sensors.camera.distortion_model.AffineTransform().distort(params, points).data

    transpiled_params = _array_to_new_backend(params, target_framework)
    transpiled_points = transpiled_kornia.geometry.vector.Vector2.from_coords(1., 2.)
    transpiled_out = transpiled_kornia.sensors.camera.distortion_model.AffineTransform().distort(transpiled_params, transpiled_points).data

    _to_numpy_and_allclose(torch_out, transpiled_out)

    params = torch.Tensor([1., 2., 3., 4.])
    points = kornia.geometry.vector.Vector2.from_coords(1., 2.)
    torch_out = kornia.sensors.camera.distortion_model.AffineTransform().undistort(params, points).data

    transpiled_params = _array_to_new_backend(params, target_framework)
    transpiled_points = transpiled_kornia.geometry.vector.Vector2.from_coords(1., 2.)
    transpiled_out = transpiled_kornia.sensors.camera.distortion_model.AffineTransform().undistort(transpiled_params, transpiled_points).data

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_Z1Projection(target_framework, mode, backend_compile):
    print("kornia.sensors.camera.projection_model.Z1Projection")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    points = kornia.geometry.vector.Vector3.from_coords(1., 2., 3.)
    torch_out = kornia.sensors.camera.projection_model.Z1Projection().project(points).data

    transpiled_points = transpiled_kornia.geometry.vector.Vector3.from_coords(1., 2., 3.)
    transpiled_out = transpiled_kornia.sensors.camera.projection_model.Z1Projection().project(transpiled_points).data

    _to_numpy_and_allclose(torch_out, transpiled_out)

    points = kornia.geometry.vector.Vector2.from_coords(1., 2.)
    torch_out = kornia.sensors.camera.projection_model.Z1Projection().unproject(points, 3).data

    transpiled_points = transpiled_kornia.geometry.vector.Vector2.from_coords(1., 2.)
    transpiled_out = transpiled_kornia.sensors.camera.projection_model.Z1Projection().unproject(transpiled_points, 3).data

    _to_numpy_and_allclose(torch_out, transpiled_out)
