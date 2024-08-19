from helpers import (
    _nest_torch_tensor_to_new_framework,
    _to_numpy_and_allclose,
    _to_numpy_and_shape_allclose,
    _test_function,
)

import ivy
import kornia
import pytest
import torch


# Tests #
# ----- #

def test_bilateral_blur(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        (3, 3),
        0.1,
        (1.5, 1.5),
    )
    trace_kwargs = {'border_type': 'reflect', 'color_distance_type': 'l1'}
    test_args = (
        torch.rand(5, 3, 5, 5),
        (5, 5),
        0.2,
        (2.0, 2.0),
    )
    test_kwargs = {'border_type': 'reflect', 'color_distance_type': 'l1'}
    _test_function(
        kornia.filters.bilateral_blur,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_blur_pool2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        3,
    )
    trace_kwargs = {'stride': 2}
    test_args = (
        torch.rand(5, 3, 8, 8),
        3,  # NOTE: changing this kernel size fails the test; also true for some of the other tests in this file
    )
    test_kwargs = {'stride': 2}
    _test_function(
        kornia.filters.blur_pool2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_box_blur(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        (3, 3),
    )
    trace_kwargs = {'border_type': 'reflect', 'separable': False}
    test_args = (
        torch.rand(5, 3, 5, 5),
        (3, 3),
    )
    test_kwargs = {'border_type': 'reflect', 'separable': False}
    _test_function(
        kornia.filters.box_blur,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_gaussian_blur2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        (3, 3),
        (1.5, 1.5),
    )
    trace_kwargs = {'border_type': 'reflect', 'separable': True}
    test_args = (
        torch.rand(5, 3, 5, 5),
        (3, 3),
        (1.5, 1.5),
    )
    test_kwargs = {'border_type': 'reflect', 'separable': True}
    _test_function(
        kornia.filters.gaussian_blur2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_guided_blur(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        torch.rand(1, 3, 5, 5),
        (3, 3),
        0.1,
    )
    trace_kwargs = {'border_type': 'reflect', 'subsample': 1}
    test_args = (
        torch.rand(5, 3, 5, 5),
        torch.rand(5, 3, 5, 5),
        (3, 3),
        0.1,
    )
    test_kwargs = {'border_type': 'reflect', 'subsample': 1}
    _test_function(
        kornia.filters.guided_blur,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_joint_bilateral_blur(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        torch.rand(1, 3, 5, 5),
        (3, 3),
        0.1,
        (1.5, 1.5),
    )
    trace_kwargs = {'border_type': 'reflect', 'color_distance_type': 'l1'}
    test_args = (
        torch.rand(4, 3, 5, 5),
        torch.rand(4, 3, 5, 5),
        (5, 5),
        0.2,
        (2.0, 2.0),
    )
    test_kwargs = {'border_type': 'reflect', 'color_distance_type': 'l1'}
    _test_function(
        kornia.filters.joint_bilateral_blur,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_max_blur_pool2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        3,
    )
    trace_kwargs = {'stride': 2, 'max_pool_size': 2, 'ceil_mode': False}
    test_args = (
        torch.rand(5, 3, 8, 8),
        3,
    )
    test_kwargs = {'stride': 2, 'max_pool_size': 2, 'ceil_mode': False}
    _test_function(
        kornia.filters.max_blur_pool2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_median_blur(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        (3, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
        (5, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.filters.median_blur,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_motion_blur(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        5,
        45.0,
        1.0,
    )
    trace_kwargs = {'border_type': 'constant', 'mode': 'nearest'}
    test_args = (
        torch.rand(5, 3, 5, 5),
        5,
        90.0,
        0.5,
    )
    test_kwargs = {'border_type': 'constant', 'mode': 'nearest'}
    _test_function(
        kornia.filters.motion_blur,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_unsharp_mask(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        (3, 3),
        (1.5, 1.5),
    )
    trace_kwargs = {'border_type': 'reflect'}
    test_args = (
        torch.rand(5, 3, 5, 5),
        (5, 5),
        (2.0, 2.0),
    )
    test_kwargs = {'border_type': 'reflect'}
    _test_function(
        kornia.filters.unsharp_mask,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_canny(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
    )
    trace_kwargs = {
        'low_threshold': 0.1,
        'high_threshold': 0.2,
        'kernel_size': (5, 5),
        'sigma': (1, 1),
        'hysteresis': True,
        'eps': 1e-6,
    }
    test_args = (
        torch.rand(5, 3, 4, 4),
    )
    test_kwargs = {
        'low_threshold': 0.2,
        'high_threshold': 0.3,
        'kernel_size': (5, 5),
        'sigma': (1, 1),
        'hysteresis': True,
        'eps': 1e-6,
    }
    _test_function(
        kornia.filters.canny,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_laplacian(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 4, 5, 5),
        3,
    )
    trace_kwargs = {
        'border_type': 'reflect',
        'normalized': True,
    }
    test_args = (
        torch.rand(5, 4, 5, 5),
        3,
    )
    test_kwargs = {
        'border_type': 'reflect',
        'normalized': True,
    }
    _test_function(
        kornia.filters.laplacian,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_sobel(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
    )
    trace_kwargs = {
        'normalized': True,
        'eps': 1e-6,
    }
    test_args = (
        torch.rand(5, 3, 4, 4),
    )
    test_kwargs = {
        'normalized': True,
        'eps': 1e-5,
    }
    _test_function(
        kornia.filters.sobel,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_spatial_gradient(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
    )
    trace_kwargs = {
        'mode': 'sobel',
        'order': 1,
        'normalized': True,
    }
    test_args = (
        torch.rand(5, 3, 4, 4),
    )
    test_kwargs = {
        'mode': 'sobel',
        'order': 1,
        'normalized': True,
    }
    _test_function(
        kornia.filters.spatial_gradient,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_spatial_gradient3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 2, 4, 4),
    )
    trace_kwargs = {
        'mode': 'diff',
        'order': 1,
    }
    test_args = (
        torch.rand(5, 4, 2, 4, 4),
    )
    test_kwargs = {
        'mode': 'diff',
        'order': 1,
    }
    _test_function(
        kornia.filters.spatial_gradient3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_filter2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5),
        torch.rand(1, 3, 3),
    )
    trace_kwargs = {'padding': 'same'}
    test_args = (
        torch.rand(2, 1, 5, 5),
        torch.rand(1, 3, 3),
    )
    test_kwargs = {'padding': 'same'}
    _test_function(
        kornia.filters.filter2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_filter2d_separable(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5),
        torch.rand(1, 3),
        torch.rand(1, 3),
    )
    trace_kwargs = {'padding': 'same'}
    test_args = (
        torch.rand(2, 1, 5, 5),
        torch.rand(1, 3),
        torch.rand(1, 3),
    )
    test_kwargs = {'padding': 'same'}
    _test_function(
        kornia.filters.filter2d_separable,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_filter3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5, 5),
        torch.rand(1, 3, 3, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 1, 5, 5, 5),
        torch.rand(1, 3, 3, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.filters.filter3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_gaussian_kernel1d(target_framework, mode, backend_compile):
    trace_args = (3, 2.5)
    trace_kwargs = {}
    test_args = (5, 1.5)
    test_kwargs = {}
    _test_function(
        kornia.filters.get_gaussian_kernel1d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_gaussian_erf_kernel1d(target_framework, mode, backend_compile):
    trace_args = (3, 2.5)
    trace_kwargs = {}
    test_args = (5, 1.5)
    test_kwargs = {}
    _test_function(
        kornia.filters.get_gaussian_erf_kernel1d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_gaussian_discrete_kernel1d(target_framework, mode, backend_compile):
    trace_args = (3, 2.5)
    trace_kwargs = {}
    test_args = (5, 1.5)
    test_kwargs = {}
    _test_function(
        kornia.filters.get_gaussian_discrete_kernel1d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_gaussian_kernel2d(target_framework, mode, backend_compile):
    trace_args = ((5, 5), (1.5, 1.5))
    trace_kwargs = {}
    test_args = ((3, 5), (1.5, 1.5))
    test_kwargs = {}
    _test_function(
        kornia.filters.get_gaussian_kernel2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_hanning_kernel1d(target_framework, mode, backend_compile):
    trace_args = (4,)
    trace_kwargs = {}
    test_args = (8,)
    test_kwargs = {}
    _test_function(
        kornia.filters.get_hanning_kernel1d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_hanning_kernel2d(target_framework, mode, backend_compile):
    trace_args = ((4, 4),)
    trace_kwargs = {}
    test_args = ((8, 8),)
    test_kwargs = {}
    _test_function(
        kornia.filters.get_hanning_kernel2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_laplacian_kernel1d(target_framework, mode, backend_compile):
    trace_args = (3,)
    trace_kwargs = {}
    test_args = (5,)
    test_kwargs = {}
    _test_function(
        kornia.filters.get_laplacian_kernel1d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_laplacian_kernel2d(target_framework, mode, backend_compile):
    trace_args = (3,)
    trace_kwargs = {}
    test_args = (5,)
    test_kwargs = {}
    _test_function(
        kornia.filters.get_laplacian_kernel2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_motion_kernel2d(target_framework, mode, backend_compile):
    trace_args = (5, 0.0)
    trace_kwargs = {'direction': 0.0, 'mode': 'nearest'}
    test_args = (3, 215.0)
    test_kwargs = {'direction': -0.5, 'mode': 'nearest'}
    _test_function(
        kornia.filters.get_motion_kernel2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_Laplacian(target_framework, mode, backend_compile):
    print("kornia.filters.Laplacian")

    if backend_compile:
        pytest.skip()

    TranspiledLaplacian = ivy.transpile(kornia.filters.Laplacian, source="torch", target=target_framework)

    x = torch.rand(2, 4, 5, 5)
    torch_out = kornia.filters.Laplacian(kernel_size=3)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledLaplacian(kernel_size=3)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_Sobel(target_framework, mode, backend_compile):
    print("kornia.filters.Sobel")

    if backend_compile:
        pytest.skip()

    TranspiledSobel = ivy.transpile(kornia.filters.Sobel, source="torch", target=target_framework)

    x = torch.rand(1, 3, 4, 4)
    torch_out = kornia.filters.Sobel()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledSobel()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_Canny(target_framework, mode, backend_compile):
    print("kornia.filters.Canny")

    if backend_compile:
        pytest.skip()

    TranspiledCanny = ivy.transpile(kornia.filters.Canny, source="torch", target=target_framework)

    x = torch.rand(5, 3, 4, 4)
    torch_out_magnitude, torch_out_edges = kornia.filters.Canny()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out_magnitude, transpiled_out_edges = TranspiledCanny()(transpiled_x)

    _to_numpy_and_allclose(torch_out_magnitude, transpiled_out_magnitude)
    _to_numpy_and_allclose(torch_out_edges, transpiled_out_edges)


def test_SpatialGradient(target_framework, mode, backend_compile):
    print("kornia.filters.SpatialGradient")

    if backend_compile:
        pytest.skip()

    TranspiledSpatialGradient = ivy.transpile(kornia.filters.SpatialGradient, source="torch", target=target_framework)

    x = torch.rand(1, 3, 4, 4)
    torch_out = kornia.filters.SpatialGradient()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledSpatialGradient()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_SpatialGradient3d(target_framework, mode, backend_compile):
    print("kornia.filters.SpatialGradient3d")

    if backend_compile:
        pytest.skip()

    TranspiledSpatialGradient3d = ivy.transpile(kornia.filters.SpatialGradient3d, source="torch", target=target_framework)

    x = torch.rand(1, 4, 2, 4, 4)
    torch_out = kornia.filters.SpatialGradient3d()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledSpatialGradient3d()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_DexiNed(target_framework, mode, backend_compile):
    print("kornia.filters.DexiNed")

    if backend_compile:
        pytest.skip()

    TranspiledDexiNed = ivy.transpile(kornia.filters.DexiNed, source="torch", target=target_framework)

    x = torch.rand(1, 3, 320, 320)
    torch_out = kornia.filters.DexiNed(pretrained=False)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledDexiNed(pretrained=False)(transpiled_x)

    _to_numpy_and_allclose(torch_out[-1], transpiled_out[-1])


def test_BilateralBlur(target_framework, mode, backend_compile):
    print("kornia.filters.BilateralBlur")

    if backend_compile:
        pytest.skip()

    TranspiledBilateralBlur = ivy.transpile(kornia.filters.BilateralBlur, source="torch", target=target_framework)

    x = torch.rand(2, 3, 5, 5)
    torch_out = kornia.filters.BilateralBlur((3, 3), 0.1, (1.5, 1.5))(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledBilateralBlur((3, 3), 0.1, (1.5, 1.5))(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_BlurPool2D(target_framework, mode, backend_compile):
    print("kornia.filters.BlurPool2D")

    if backend_compile:
        pytest.skip()

    TranspiledBlurPool2D = ivy.transpile(kornia.filters.BlurPool2D, source="torch", target=target_framework)

    x = torch.eye(5)[None, None]
    torch_out = kornia.filters.BlurPool2D(kernel_size=3, stride=2)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledBlurPool2D(kernel_size=3, stride=2)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_BoxBlur(target_framework, mode, backend_compile):
    print("kornia.filters.BoxBlur")

    if backend_compile:
        pytest.skip()

    TranspiledBoxBlur = ivy.transpile(kornia.filters.BoxBlur, source="torch", target=target_framework)

    x = torch.rand(2, 3, 5, 5)
    torch_out = kornia.filters.BoxBlur(kernel_size=(3, 3))(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledBoxBlur(kernel_size=(3, 3))(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_MaxBlurPool2D(target_framework, mode, backend_compile):
    print("kornia.filters.MaxBlurPool2D")

    if backend_compile:
        pytest.skip()

    TranspiledMaxBlurPool2D = ivy.transpile(kornia.filters.MaxBlurPool2D, source="torch", target=target_framework)

    x = torch.eye(5)[None, None]
    torch_out = kornia.filters.MaxBlurPool2D(kernel_size=3, stride=2, max_pool_size=2, ceil_mode=False)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledMaxBlurPool2D(kernel_size=3, stride=2, max_pool_size=2, ceil_mode=False)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_MedianBlur(target_framework, mode, backend_compile):
    print("kornia.filters.MedianBlur")

    if backend_compile:
        pytest.skip()

    TranspiledMedianBlur = ivy.transpile(kornia.filters.MedianBlur, source="torch", target=target_framework)

    x = torch.rand(2, 3, 5, 5)
    torch_out = kornia.filters.MedianBlur(kernel_size=(3, 3))(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledMedianBlur(kernel_size=(3, 3))(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_GaussianBlur2d(target_framework, mode, backend_compile):
    print("kornia.filters.GaussianBlur2d")

    if backend_compile:
        pytest.skip()

    TranspiledGaussianBlur2d = ivy.transpile(kornia.filters.GaussianBlur2d, source="torch", target=target_framework)

    x = torch.rand(2, 3, 5, 5)
    torch_out = kornia.filters.GaussianBlur2d((3, 3), (1.5, 1.5))(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledGaussianBlur2d((3, 3), (1.5, 1.5))(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_GuidedBlur(target_framework, mode, backend_compile):
    print("kornia.filters.GuidedBlur")

    if backend_compile:
        pytest.skip()

    TranspiledGuidedBlur = ivy.transpile(kornia.filters.GuidedBlur, source="torch", target=target_framework)

    guidance = torch.rand(2, 3, 5, 5)
    x = torch.rand(2, 3, 5, 5)
    torch_out = kornia.filters.GuidedBlur(3, 0.1)(guidance, x)

    transpiled_guidance = _nest_torch_tensor_to_new_framework(guidance, target_framework)
    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledGuidedBlur(3, 0.1)(transpiled_guidance, transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_JointBilateralBlur(target_framework, mode, backend_compile):
    print("kornia.filters.JointBilateralBlur")

    if backend_compile:
        pytest.skip()

    TranspiledJointBilateralBlur = ivy.transpile(kornia.filters.JointBilateralBlur, source="torch", target=target_framework)

    guidance = torch.rand(2, 3, 5, 5)
    x = torch.rand(2, 3, 5, 5)
    torch_out = kornia.filters.JointBilateralBlur((3, 3), 0.1, (1.5, 1.5))(guidance, x)

    transpiled_guidance = _nest_torch_tensor_to_new_framework(guidance, target_framework)
    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledJointBilateralBlur((3, 3), 0.1, (1.5, 1.5))(transpiled_guidance, transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_MotionBlur(target_framework, mode, backend_compile):
    print("kornia.filters.MotionBlur")

    if backend_compile:
        pytest.skip()

    TranspiledMotionBlur = ivy.transpile(kornia.filters.MotionBlur, source="torch", target=target_framework)

    x = torch.rand(2, 3, 5, 7)
    torch_out = kornia.filters.MotionBlur(3, 35., 0.5)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledMotionBlur(3, 35., 0.5)(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_UnsharpMask(target_framework, mode, backend_compile):
    print("kornia.filters.UnsharpMask")

    if backend_compile:
        pytest.skip()

    TranspiledUnsharpMask = ivy.transpile(kornia.filters.UnsharpMask, source="torch", target=target_framework)

    x = torch.rand(2, 3, 5, 5)
    torch_out = kornia.filters.UnsharpMask((3, 3), (1.5, 1.5))(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledUnsharpMask((3, 3), (1.5, 1.5))(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_InRange(target_framework, mode, backend_compile):
    print("kornia.filters.InRange")

    if backend_compile:
        pytest.skip()

    TranspiledInRange = ivy.transpile(kornia.filters.InRange, source="torch", target=target_framework)

    x = torch.rand(1, 3, 3, 3)
    torch_out = kornia.filters.InRange((0.2, 0.3, 0.4), (0.8, 0.9, 1.0), return_mask=True)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledInRange((0.2, 0.3, 0.4), (0.8, 0.9, 1.0), return_mask=True)(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)
