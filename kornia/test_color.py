from helpers import (
    _array_to_new_backend,
    _check_allclose,
    _nest_array_to_numpy,
    _nest_torch_tensor_to_new_framework,
    _test_function,
    _to_numpy_and_allclose,
)
import ivy
import kornia
import pytest
import torch


# Helpers #
# ------- #

def _test_color_class(
    cls,
    args,
    target,
    backend_compile=False,
    tolerance=1e-3,
    init_args=(),
):
    print(f"{cls.__module__}.{cls.__name__}")

    if backend_compile:
        pytest.skip()

    transpiled_cls = ivy.transpile(cls, source="torch", target=target)

    torch_obj = cls(*init_args)
    transpiled_obj = transpiled_cls(*init_args)

    torch_out = torch_obj(*args)
    transpile_args = _nest_torch_tensor_to_new_framework(args, target)
    transpiled_out = transpiled_obj(*transpile_args)

    orig_np = _nest_array_to_numpy(torch_out)
    graph_np = _nest_array_to_numpy(transpiled_out)

    _check_allclose(orig_np, graph_np, tolerance=tolerance)


# Tests #
# ----- #

def test_rgb_to_grayscale(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.rgb_to_grayscale,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_bgr_to_grayscale(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.bgr_to_grayscale,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_grayscale_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.grayscale_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_rgb_to_bgr(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 3, 5, 5),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3, 5, 5),)
    test_kwargs = {}
    _test_function(
        kornia.color.rgb_to_bgr,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_bgr_to_rgb(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 3, 5, 5),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3, 5, 5),)
    test_kwargs = {}
    _test_function(
        kornia.color.bgr_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_rgb_to_linear_rgb(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 3, 5, 5),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3, 5, 5),)
    test_kwargs = {}
    _test_function(
        kornia.color.rgb_to_linear_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_linear_rgb_to_rgb(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 3, 5, 5),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3, 5, 5),)
    test_kwargs = {}
    _test_function(
        kornia.color.linear_rgb_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_bgr_to_rgba(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
        1.0,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 4),
        0.5,
    )
    test_kwargs = {}
    _test_function(
        kornia.color.bgr_to_rgba,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_rgb_to_rgba(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
        1.0,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 4),
        0.5,
    )
    test_kwargs = {}
    _test_function(
        kornia.color.rgb_to_rgba,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_rgba_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 4, 4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.rgba_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_rgba_to_bgr(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 4, 4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.rgba_to_bgr,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_rgb_to_hls(target_framework, mode, backend_compile):
    # Note: We test this function with requires_grad=True,
    # because otherwise we simply get an empty_like tensor
    # with garbage values on each run leading to test failures
    trace_args = (
        torch.rand(1, 3, 4, 5).requires_grad_(True),
    )
    trace_kwargs = {'eps': 1e-8}
    test_args = (
        torch.rand(5, 3, 4, 5).requires_grad_(True),
    )
    test_kwargs = {'eps': 1e-8}
    _test_function(
        kornia.color.rgb_to_hls,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_hls_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.hls_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_rgb_to_hsv(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {'eps': 1e-8}
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {'eps': 1e-2}
    _test_function(
        kornia.color.rgb_to_hsv,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_hsv_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.hsv_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_rgb_to_luv(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {
        'eps': 1e-12
    }
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {
        'eps': 1e-12
    }
    _test_function(
        kornia.color.rgb_to_luv,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_luv_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {'eps': 1e-12}
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {'eps': 1e-12}
    _test_function(
        kornia.color.luv_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_rgb_to_lab(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.rgb_to_lab,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_lab_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {'clip': True}
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {'clip': True}
    _test_function(
        kornia.color.lab_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_rgb_to_ycbcr(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.rgb_to_ycbcr,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_ycbcr_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.ycbcr_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_rgb_to_yuv(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.rgb_to_yuv,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_yuv_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.yuv_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_rgb_to_yuv420(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 6),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 6),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.rgb_to_yuv420,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_yuv420_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 4, 6),
        torch.rand(1, 2, 2, 3) * 2.0 - 0.5,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 4, 6),
        torch.rand(5, 2, 2, 3) * 2.0 - 0.5,
    )
    test_kwargs = {}
    _test_function(
        kornia.color.yuv420_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_rgb_to_yuv422(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 6, 8),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 6, 8),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.rgb_to_yuv422,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_yuv422_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2, 6),
        torch.rand(1, 2, 2, 3) - 0.5,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 2, 6),
        torch.rand(5, 2, 2, 3) - 0.5,
    )
    test_kwargs = {}
    _test_function(
        kornia.color.yuv422_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_rgb_to_xyz(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.rgb_to_xyz,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_xyz_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.xyz_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_rgb_to_raw(target_framework, mode, backend_compile):
    print("kornia.color.rgb_to_raw")

    TranspiledCFA = ivy.transpile(kornia.color.CFA, source="torch", target=target_framework)
    transpiled_rgb_to_raw = ivy.transpile(kornia.color.rgb_to_raw, source="torch", target=target_framework)

    torch_x = torch.rand(5, 3, 5, 5)
    transpiled_x = _array_to_new_backend(torch_x, target_framework)

    torch_out = kornia.color.rgb_to_raw(torch_x, kornia.color.CFA.BG)
    transpiled_out = transpiled_rgb_to_raw(transpiled_x, TranspiledCFA.BG)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_raw_to_rgb(target_framework, mode, backend_compile):
    print("kornia.color.raw_to_rgb")

    transpiled_raw_to_rgb = ivy.transpile(kornia.color.raw_to_rgb, source="torch", target=target_framework)
    TranspiledCFA = ivy.transpile(kornia.color.CFA, source="torch", target=target_framework)

    torch_x = torch.rand(5, 1, 4, 6)
    transpiled_x = _array_to_new_backend(torch_x, target_framework)

    torch_out = kornia.color.raw_to_rgb(torch_x, kornia.color.CFA.RG)
    transpiled_out = transpiled_raw_to_rgb(transpiled_x, TranspiledCFA.RG)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_raw_to_rgb_2x2_downscaled(target_framework, mode, backend_compile):
    print("kornia.color.raw_to_rgb_2x2_downscaled")

    transpiled_raw_to_rgb_2x2_downscaled = ivy.transpile(kornia.color.raw_to_rgb_2x2_downscaled, source="torch", target=target_framework)
    TranspiledCFA = ivy.transpile(kornia.color.CFA, source="torch", target=target_framework)

    torch_x = torch.rand(5, 1, 4, 6)
    transpiled_x = _array_to_new_backend(torch_x, target_framework)

    torch_out = kornia.color.raw_to_rgb_2x2_downscaled(torch_x, kornia.color.CFA.RG)
    transpiled_out = transpiled_raw_to_rgb_2x2_downscaled(transpiled_x, TranspiledCFA.RG)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_sepia(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
    )
    trace_kwargs = {
        'rescale': True,
        'eps': 1e-6,
    }
    test_args = (
        torch.rand(5, 3, 4, 4),
    )
    test_kwargs = {
        'rescale': True,
        'eps': 1e-6,
    }
    _test_function(
        kornia.color.sepia,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_GrayscaleToRgb(target_framework, mode, backend_compile):
    args = (
        torch.rand(5, 1, 4, 4),
    )
    _test_color_class(
        kornia.color.GrayscaleToRgb,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_RgbToGrayscale(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 5),
    )
    _test_color_class(
        kornia.color.RgbToGrayscale,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_BgrToGrayscale(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 5),
    )
    _test_color_class(
        kornia.color.BgrToGrayscale,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_RgbToBgr(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 5),
    )
    _test_color_class(
        kornia.color.RgbToBgr,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_BgrToRgb(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 5),
    )
    _test_color_class(
        kornia.color.BgrToRgb,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_LinearRgbToRgb(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 5),
    )
    _test_color_class(
        kornia.color.LinearRgbToRgb,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_RgbToLinearRgb(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 5),
    )
    _test_color_class(
        kornia.color.RgbToLinearRgb,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_RgbToRgba(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 5),
    )
    _test_color_class(
        kornia.color.RgbToRgba,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        init_args=(1.,),
    )


def test_BgrToRgba(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 5),
    )
    _test_color_class(
        kornia.color.BgrToRgba,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        init_args=(1.,),
    )


def test_RgbaToRgb(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 4, 4, 5),
    )
    _test_color_class(
        kornia.color.RgbaToRgb,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_RgbaToBgr(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 4, 4, 5),
    )
    _test_color_class(
        kornia.color.RgbaToBgr,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_RgbToHls(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 5),
    )
    _test_color_class(
        kornia.color.RgbToHls,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_HlsToRgb(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 5),
    )
    _test_color_class(
        kornia.color.HlsToRgb,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_RgbToHsv(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 5),
    )
    _test_color_class(
        kornia.color.RgbToHsv,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_HsvToRgb(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 5),
    )
    _test_color_class(
        kornia.color.HsvToRgb,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_RgbToLuv(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 5),
    )
    _test_color_class(
        kornia.color.RgbToLuv,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_LuvToRgb(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 5),
    )
    _test_color_class(
        kornia.color.LuvToRgb,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_RgbToLab(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 5),
    )
    _test_color_class(
        kornia.color.RgbToLab,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_LabToRgb(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 5),
    )
    _test_color_class(
        kornia.color.LabToRgb,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_YcbcrToRgb(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 5),
    )
    _test_color_class(
        kornia.color.YcbcrToRgb,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_RgbToYcbcr(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 5),
    )
    _test_color_class(
        kornia.color.RgbToYcbcr,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_RgbToYuv(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 5),
    )
    _test_color_class(
        kornia.color.RgbToYuv,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_YuvToRgb(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 5),
    )
    _test_color_class(
        kornia.color.YuvToRgb,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_RgbToYuv420(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 6),
    )
    _test_color_class(
        kornia.color.RgbToYuv420,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_Yuv420ToRgb(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 1, 4, 6),
        torch.rand(2, 2, 2, 3),
    )
    _test_color_class(
        kornia.color.Yuv420ToRgb,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_RgbToYuv422(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 6),
    )
    _test_color_class(
        kornia.color.RgbToYuv422,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_Yuv422ToRgb(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 1, 4, 6),
        torch.rand(2, 2, 4, 3),
    )
    _test_color_class(
        kornia.color.Yuv422ToRgb,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_RgbToXyz(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 5),
    )
    _test_color_class(
        kornia.color.RgbToXyz,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_XyzToRgb(target_framework, mode, backend_compile):
    args = (
        torch.rand(2, 3, 4, 5),
    )
    _test_color_class(
        kornia.color.XyzToRgb,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_RawToRgb(target_framework, mode, backend_compile):
    print("kornia.color.RawToRgb")

    transpiled_RawToRgb = ivy.transpile(kornia.color.RawToRgb, source="torch", target=target_framework)
    TranspiledCFA = ivy.transpile(kornia.color.CFA, source="torch", target=target_framework)

    torch_x = torch.rand(2, 1, 4, 6)
    transpiled_x = _array_to_new_backend(torch_x, target_framework)

    torch_out = kornia.color.RawToRgb(kornia.color.CFA.RG)(torch_x)
    transpiled_out = transpiled_RawToRgb(TranspiledCFA.RG)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_RgbToRaw(target_framework, mode, backend_compile):
    print("kornia.color.RgbToRaw")

    transpiled_RgbToRaw = ivy.transpile(kornia.color.RgbToRaw, source="torch", target=target_framework)
    TranspiledCFA = ivy.transpile(kornia.color.CFA, source="torch", target=target_framework)

    torch_x = torch.rand(2, 3, 4, 6)
    transpiled_x = _array_to_new_backend(torch_x, target_framework)

    torch_out = kornia.color.RgbToRaw(kornia.color.CFA.GB)(torch_x)
    transpiled_out = transpiled_RgbToRaw(TranspiledCFA.GB)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_RawToRgb2x2Downscaled(target_framework, mode, backend_compile):
    print("kornia.color.RawToRgb2x2Downscaled")

    transpiled_RawToRgb2x2Downscaled = ivy.transpile(kornia.color.RawToRgb2x2Downscaled, source="torch", target=target_framework)
    TranspiledCFA = ivy.transpile(kornia.color.CFA, source="torch", target=target_framework)

    torch_x = torch.rand(2, 1, 4, 6)
    transpiled_x = _array_to_new_backend(torch_x, target_framework)

    torch_out = kornia.color.RawToRgb2x2Downscaled(kornia.color.CFA.RG)(torch_x)
    transpiled_out = transpiled_RawToRgb2x2Downscaled(TranspiledCFA.RG)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_Sepia(target_framework, mode, backend_compile):
    args = (
        torch.ones(3, 1, 1),
    )
    _test_color_class(
        kornia.color.Sepia,
        args,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
    )


def test_CFA(target_framework, mode, backend_compile):
    print("kornia.color.CFA")

    if backend_compile:
        pytest.skip()

    TranspiledCFA = ivy.transpile(kornia.color.CFA, source="torch", target=target_framework)

    assert TranspiledCFA(0).value == 0
    assert TranspiledCFA(0).name == "BG"
    assert TranspiledCFA(1).name == "GB"
    assert TranspiledCFA(2).name == "RG"
    assert TranspiledCFA(3).name == "GR"
    assert TranspiledCFA.BG.value == 0
    assert TranspiledCFA.GB.value == 1
    assert TranspiledCFA.RG.value == 2
    assert TranspiledCFA.GR.value == 3


def test_ColorMap(target_framework, mode, backend_compile):
    print("kornia.color.ColorMap")

    if backend_compile:
        pytest.skip()

    TranspiledColorMap = ivy.transpile(kornia.color.ColorMap, source="torch", target=target_framework)

    torch_out = kornia.color.ColorMap(base='viridis', num_colors=8).colors
    transpiled_out = TranspiledColorMap(base='viridis', num_colors=8).colors

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)
    _check_allclose(orig_np, transpiled_np)


def test_apply_colormap(target_framework, mode, backend_compile):
    print("kornia.color.ColorMap")

    if backend_compile:
        pytest.skip()

    TranspiledColorMapType = ivy.transpile(kornia.color.ColorMapType, source="torch", target=target_framework)
    TranspiledColorMap = ivy.transpile(kornia.color.ColorMap, source="torch", target=target_framework)
    transpiled_apply_colormap = ivy.transpile(kornia.color.apply_colormap, source="torch", target=target_framework)

    torch_x = torch.tensor([[[0, 1, 2], [15, 25, 33], [128, 158, 188]]])
    transpiled_x = _array_to_new_backend(torch_x, target_framework)

    colormap = kornia.color.ColorMap(base=kornia.color.ColorMapType.autumn)
    torch_out = kornia.color.apply_colormap(torch_x, colormap)

    colormap = TranspiledColorMap(base=TranspiledColorMapType.autumn)
    transpiled_out = transpiled_apply_colormap(transpiled_x, colormap)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)
    _check_allclose(orig_np, transpiled_np)
