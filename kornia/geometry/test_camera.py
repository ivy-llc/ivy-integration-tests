from helpers import (
    _check_allclose,
    _nest_array_to_numpy,
    _nest_torch_tensor_to_new_framework,
    _test_function,
)

import ivy
import kornia
import pytest
import torch


# Tests #
# ----- #

def test_project_points_z1(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.project_points_z1,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_unproject_points_z1(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2),
        torch.rand(1),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2),
        torch.rand(1),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.unproject_points_z1,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_dx_project_points_z1(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.dx_project_points_z1,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_project_points_orthographic(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.project_points_orthographic,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_unproject_points_orthographic(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 2),
        torch.rand(2, 1),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 2),
        torch.rand(5, 1),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.unproject_points_orthographic,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_dx_project_points_orthographic(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.dx_project_points_orthographic,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_distort_points_affine(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 2),
        torch.rand(2, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 2),
        torch.rand(5, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.distort_points_affine,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_undistort_points_affine(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 2),
        torch.rand(2, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 2),
        torch.rand(5, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.undistort_points_affine,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_dx_distort_points_affine(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 2),
        torch.rand(2, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 2),
        torch.rand(5, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.dx_distort_points_affine,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_distort_points_kannala_brandt(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 2),
        torch.rand(2, 8),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 2),
        torch.rand(5, 8),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.distort_points_kannala_brandt,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_undistort_points_kannala_brandt(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 2),
        torch.rand(2, 8),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 2),
        torch.rand(5, 8),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.undistort_points_kannala_brandt,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_dx_distort_points_kannala_brandt(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 2),
        torch.rand(2, 8),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 2),
        torch.rand(5, 8),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.dx_distort_points_kannala_brandt,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_cam2pixel(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 3, 3, 3),
        torch.rand(2, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 3, 3),
        torch.rand(5, 4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.pinhole.cam2pixel,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_pixel2cam(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 1, 3, 3),
        torch.rand(2, 4, 4),
        torch.rand(2, 3, 3, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 3, 3),
        torch.rand(5, 4, 4),
        torch.rand(5, 3, 3, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.pinhole.pixel2cam,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_project_points(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 3),
        torch.eye(3)[None].repeat(2, 1, 1),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3),
        torch.eye(3)[None].repeat(5, 1, 1),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.perspective.project_points,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_unproject_points(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 2),
        torch.ones(2, 1),
        torch.eye(3)[None].repeat(2, 1, 1),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 2),
        torch.ones(5, 1),
        torch.eye(3)[None].repeat(5, 1, 1),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.perspective.unproject_points,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_reproject_disparity_to_3D(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 2, 2, 1),
        torch.rand(1, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 10, 10, 1),
        torch.rand(5, 4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.stereo.reproject_disparity_to_3D,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_PinholeCamera(target_framework, mode, backend_compile):
    print("kornia.geometry.camera.pinhole.PinholeCamera")

    if backend_compile:
        pytest.skip()

    TranspiledPinholeCamera = ivy.transpile(kornia.geometry.camera.pinhole.PinholeCamera, source="torch", target=target_framework)

    torch_init_args = (
        torch.eye(4)[None],
        torch.eye(4)[None],
        torch.ones(1),
        torch.ones(1),
    )
    torch_call_args = (
        torch.rand(1, 3),
    )
    transpiled_init_args = _nest_torch_tensor_to_new_framework(torch_init_args, target_framework)
    transpiled_call_args = _nest_torch_tensor_to_new_framework(torch_call_args, target_framework)

    torch_pinhole = kornia.geometry.camera.PinholeCamera(*torch_init_args)
    torch_out = torch_pinhole.project(*torch_call_args)

    transpiled_pinhole = TranspiledPinholeCamera(*transpiled_init_args)
    transpiled_out = transpiled_pinhole.project(*transpiled_call_args)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)
    _check_allclose(orig_np, transpiled_np)

    # TODO: test the other methods/attributes of this class


def test_StereoCamera(target_framework, mode, backend_compile):
    print("kornia.geometry.camera.stereo.StereoCamera")

    if backend_compile:
        pytest.skip()

    TranspiledStereoCamera = ivy.transpile(kornia.geometry.camera.stereo.StereoCamera, source="torch", target=target_framework)

    torch_init_args = (
        -torch.ones(2, 3, 4),
        -torch.ones(2, 3, 4),
    )
    transpiled_init_args = _nest_torch_tensor_to_new_framework(torch_init_args, target_framework)

    torch_camera = kornia.geometry.camera.stereo.StereoCamera(*torch_init_args)
    transpiled_camera = TranspiledStereoCamera(*transpiled_init_args)

    assert transpiled_camera.batch_size == torch_camera.batch_size
