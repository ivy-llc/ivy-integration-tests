from helpers import (
    _check_allclose,
    _check_shape_allclose,
    _nest_array_to_numpy,
    _nest_torch_tensor_to_new_framework,
    _test_function,
)

import ivy
import kornia
import pytest
import torch


# Helpers #
# ------- #

def _to_numpy_and_allclose(torch_x, transpiled_x, tolerance=1e-3):
    orig_data = _nest_array_to_numpy(torch_x)
    transpiled_data = _nest_array_to_numpy(transpiled_x)
    _check_allclose(orig_data, transpiled_data, tolerance=tolerance)


def _to_numpy_and_shape_allclose(torch_x, transpiled_x, tolerance=1e-3):
    orig_data = _nest_array_to_numpy(torch_x)
    transpiled_data = _nest_array_to_numpy(transpiled_x)
    _check_shape_allclose(orig_data, transpiled_data, tolerance=tolerance) 


# Tests #
# ----- #

def test_conv_soft_argmax2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(20, 16, 50, 32),
    )
    trace_kwargs = {
        'kernel_size': (3, 3),
        'stride': (1, 1),
        'padding': (1, 1),
        'temperature': torch.tensor(1.0),
        'normalized_coordinates': True,
        'eps': 1e-8,
        'output_value': True,
    }
    test_args = (
        torch.rand(10, 16, 50, 32),
    )
    test_kwargs = {
        'kernel_size': (3, 3),
        'stride': (1, 1),
        'padding': (1, 1),
        'temperature': torch.tensor(0.5),
        'normalized_coordinates': True,
        'eps': 1e-8,
        'output_value': True,
    }
    _test_function(
        kornia.geometry.subpix.conv_soft_argmax2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_conv_soft_argmax3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(20, 16, 3, 50, 32),
    )
    trace_kwargs = {
        'kernel_size': (3, 3, 3),
        'stride': (1, 1, 1),
        'padding': (1, 1, 1),
        'temperature': torch.tensor(1.0),
        'normalized_coordinates': False,
        'eps': 1e-8,
        'output_value': True,
        'strict_maxima_bonus': 0.0,
    }
    test_args = (
        torch.rand(10, 16, 5, 50, 32),
    )
    test_kwargs = {
        'kernel_size': (3, 3, 3),
        'stride': (1, 1, 1),
        'padding': (1, 1, 1),
        'temperature': torch.tensor(0.5),
        'normalized_coordinates': False,
        'eps': 1e-8,
        'output_value': True,
        'strict_maxima_bonus': 0.0,
    }
    _test_function(
        kornia.geometry.subpix.conv_soft_argmax3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_conv_quad_interp3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(20, 16, 3, 50, 32),
    )
    trace_kwargs = {
        'strict_maxima_bonus': 10.0,
        'eps': 1e-7,
    }
    test_args = (
        torch.rand(10, 16, 5, 50, 32),
    )
    test_kwargs = {
        'strict_maxima_bonus': 5.0,
        'eps': 1e-7,
    }
    _test_function(
        kornia.geometry.subpix.conv_quad_interp3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_spatial_softmax2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5),
    )
    trace_kwargs = {
        'temperature': torch.tensor(1.0),
    }
    test_args = (
        torch.rand(10, 1, 5, 5),
    )
    test_kwargs = {
        'temperature': torch.tensor(0.5),
    }
    _test_function(
        kornia.geometry.subpix.spatial_softmax2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_spatial_expectation2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5),
    )
    trace_kwargs = {
        'normalized_coordinates': False,
    }
    test_args = (
        torch.rand(10, 1, 5, 5),
    )
    test_kwargs = {
        'normalized_coordinates': False,
    }
    _test_function(
        kornia.geometry.subpix.spatial_expectation2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_spatial_soft_argmax2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5),
    )
    trace_kwargs = {
        'temperature': torch.tensor(1.0),
        'normalized_coordinates': True,
    }
    test_args = (
        torch.rand(10, 1, 5, 5),
    )
    test_kwargs = {
        'temperature': torch.tensor(0.5),
        'normalized_coordinates': True,
    }
    _test_function(
        kornia.geometry.subpix.spatial_soft_argmax2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_render_gaussian2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[1.0, 1.0]]),
        torch.tensor([[1.0, 1.0]]),
        (5, 5),
    )
    trace_kwargs = {
        'normalized_coordinates': False,
    }
    test_args = (
        torch.tensor([[2.0, 2.0]]),
        torch.tensor([[0.5, 0.5]]),
        (10, 10),
    )
    test_kwargs = {
        'normalized_coordinates': False,
    }
    _test_function(
        kornia.geometry.subpix.render_gaussian2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_nms2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5),
        (3, 3),
    )
    trace_kwargs = {
        'mask_only': False,
    }
    test_args = (
        torch.rand(10, 1, 5, 5),
        (3, 3),
    )
    test_kwargs = {
        'mask_only': False,
    }
    _test_function(
        kornia.geometry.subpix.nms2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_nms3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5, 5),
        (3, 3, 3),
    )
    trace_kwargs = {
        'mask_only': False,
    }
    test_args = (
        torch.rand(10, 1, 5, 5, 5),
        (3, 3, 3),
    )
    test_kwargs = {
        'mask_only': False,
    }
    _test_function(
        kornia.geometry.subpix.nms3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_SpatialSoftArgmax2d(target_framework, mode, backend_compile):
    print("kornia.geometry.subpix.SpatialSoftArgmax2d")

    if backend_compile:
        pytest.skip()

    TranspiledSpatialSoftArgmax2d = ivy.transpile(
        kornia.geometry.subpix.SpatialSoftArgmax2d, source="torch", target=target_framework
    )

    spatial_soft_argmax2d = kornia.geometry.subpix.SpatialSoftArgmax2d()
    transpiled_spatial_soft_argmax2d = TranspiledSpatialSoftArgmax2d()    

    heatmap = torch.randn(10, 3, 5, 5, requires_grad=True)
    transpiled_heatmap = _nest_torch_tensor_to_new_framework(heatmap, target_framework)

    torch_output = spatial_soft_argmax2d(heatmap)
    transpiled_output = transpiled_spatial_soft_argmax2d(transpiled_heatmap)

    _to_numpy_and_allclose(torch_output, transpiled_output)


def test_ConvSoftArgmax2d(target_framework, mode, backend_compile):
    print("kornia.geometry.subpix.ConvSoftArgmax2d")

    if backend_compile:
        pytest.skip()

    TranspiledConvSoftArgmax2d = ivy.transpile(
        kornia.geometry.subpix.ConvSoftArgmax2d, source="torch", target=target_framework
    )

    conv_soft_argmax2d = kornia.geometry.subpix.ConvSoftArgmax2d()
    transpiled_conv_soft_argmax2d = TranspiledConvSoftArgmax2d()

    heatmap = torch.randn(1, 1, 5, 5, requires_grad=True)
    transpiled_heatmap = _nest_torch_tensor_to_new_framework(heatmap, target_framework)

    torch_output = conv_soft_argmax2d(heatmap)
    transpiled_output = transpiled_conv_soft_argmax2d(transpiled_heatmap)

    _to_numpy_and_allclose(torch_output, transpiled_output)


def test_ConvSoftArgmax3d(target_framework, mode, backend_compile):
    print("kornia.geometry.subpix.ConvSoftArgmax3d")

    if backend_compile:
        pytest.skip()

    TranspiledConvSoftArgmax3d = ivy.transpile(
        kornia.geometry.subpix.ConvSoftArgmax3d, source="torch", target=target_framework
    )

    conv_soft_argmax3d = kornia.geometry.subpix.ConvSoftArgmax3d()
    transpiled_conv_soft_argmax3d = TranspiledConvSoftArgmax3d()

    heatmap = torch.randn(1, 1, 3, 5, 5, requires_grad=True)
    transpiled_heatmap = _nest_torch_tensor_to_new_framework(heatmap, target_framework)

    torch_output = conv_soft_argmax3d(heatmap)
    transpiled_output = transpiled_conv_soft_argmax3d(transpiled_heatmap)

    _to_numpy_and_allclose(torch_output, transpiled_output)


def test_ConvQuadInterp3d(target_framework, mode, backend_compile):
    print("kornia.geometry.subpix.ConvQuadInterp3d")

    if backend_compile:
        pytest.skip()

    TranspiledConvQuadInterp3d = ivy.transpile(
        kornia.geometry.subpix.ConvQuadInterp3d, source="torch", target=target_framework
    )

    conv_quad_interp3d = kornia.geometry.subpix.ConvQuadInterp3d()
    transpiled_conv_quad_interp3d = TranspiledConvQuadInterp3d()

    heatmap = torch.randn(1, 1, 3, 5, 5, requires_grad=True)
    transpiled_heatmap = _nest_torch_tensor_to_new_framework(heatmap, target_framework)

    torch_output = conv_quad_interp3d(heatmap)
    transpiled_output = transpiled_conv_quad_interp3d(transpiled_heatmap)

    _to_numpy_and_allclose(torch_output, transpiled_output)


def test_NonMaximaSuppression2d(target_framework, mode, backend_compile):
    print("kornia.geometry.subpix.NonMaximaSuppression2d")

    if backend_compile:
        pytest.skip()

    TranspiledNonMaximaSuppression2d = ivy.transpile(
        kornia.geometry.subpix.NonMaximaSuppression2d, source="torch", target=target_framework
    )

    non_maxima_suppression2d = kornia.geometry.subpix.NonMaximaSuppression2d(kernel_size=(3, 3))
    transpiled_non_maxima_suppression2d = TranspiledNonMaximaSuppression2d(kernel_size=(3, 3))    

    heatmap = torch.randn(1, 1, 5, 5, requires_grad=True)
    transpiled_heatmap = _nest_torch_tensor_to_new_framework(heatmap, target_framework)

    torch_output = non_maxima_suppression2d(heatmap)
    transpiled_output = transpiled_non_maxima_suppression2d(transpiled_heatmap)

    _to_numpy_and_allclose(torch_output, transpiled_output)


def test_NonMaximaSuppression3d(target_framework, mode, backend_compile):
    print("kornia.geometry.subpix.NonMaximaSuppression3d")

    if backend_compile:
        pytest.skip()

    TranspiledNonMaximaSuppression3d = ivy.transpile(
        kornia.geometry.subpix.NonMaximaSuppression3d, source="torch", target=target_framework
    )

    non_maxima_suppression3d = kornia.geometry.subpix.NonMaximaSuppression3d(kernel_size=(3, 3, 3))
    transpiled_non_maxima_suppression3d = TranspiledNonMaximaSuppression3d(kernel_size=(3, 3, 3))

    heatmap = torch.randn(1, 1, 3, 5, 5, requires_grad=True)
    transpiled_heatmap = _nest_torch_tensor_to_new_framework(heatmap, target_framework)

    torch_output = non_maxima_suppression3d(heatmap)
    transpiled_output = transpiled_non_maxima_suppression3d(transpiled_heatmap)

    _to_numpy_and_allclose(torch_output, transpiled_output)
