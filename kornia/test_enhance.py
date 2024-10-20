from helpers import (
    _nest_torch_tensor_to_new_framework,
    _to_numpy_and_allclose,
    _test_function,
)
import ivy
import kornia
import pytest
import torch

# Tests #
# ----- #


def test_add_weighted(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5),
        0.5,
        torch.rand(1, 1, 5, 5),
        0.5,
        1.0,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 5, 5),
        0.7,
        torch.rand(5, 1, 5, 5),
        0.8,
        0.8,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.add_weighted,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_adjust_brightness(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2, 2),
        1.0,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 2, 2),
        1.3,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.adjust_brightness,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_adjust_contrast(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2, 2),
        2.0,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 2, 2),
        0.5,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.adjust_contrast,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_adjust_contrast_with_mean_subtraction(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2, 2),
        2.0,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 2, 2),
        0.5,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.adjust_contrast_with_mean_subtraction,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_adjust_gamma(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2, 2),
        2.2,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 2, 2),
        0.4,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.adjust_gamma,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_adjust_hue(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 2, 2),
        0.5,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 2, 2),
        -0.2,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.adjust_hue,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_adjust_saturation(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 2, 2),
        0.5,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 2, 2),
        1.5,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.adjust_saturation,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_adjust_sigmoid(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2, 2),
        0.5,
        0.1,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 2, 2),
        0.7,
        0.05,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.adjust_sigmoid,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_adjust_log(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2, 2),
        0.5,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 2, 2),
        1.2,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.adjust_log,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_invert(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 1, 2, 2),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 1, 2, 2),)
    test_kwargs = {}
    _test_function(
        kornia.enhance.invert,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_posterize(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2, 2),
        3,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 2, 2),
        4,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.posterize,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e0, # numerical differences b/w PT and JAX
        mode=mode,
    )


def test_sharpness(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5),
        0.5,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 5, 5),
        1.0,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.sharpness,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_solarize(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2, 2),
        0.5,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 2, 2),
        0.7,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.solarize,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


# TODO: DCF handling: uses a loop to iterate over the batch dim(https://github.com/kornia/kornia/blob/bdeac07e1edc26863a9c8d0826dc202974fd850a/kornia/enhance/adjust.py#L946)
def test_equalize(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 2, 3, 3),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 2, 3, 3),)
    test_kwargs = {}
    _test_function(
        kornia.enhance.equalize,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
        skip=True,
    )


# TODO: DCF handling: uses a conditional to modify the shape of the tensor(https://github.com/kornia/kornia/blob/bdeac07e1edc26863a9c8d0826dc202974fd850a/kornia/enhance/equalization.py#L39)
def test_equalize_clahe(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 10, 20),)
    trace_kwargs = {
        "clip_limit": 40.0,
        "grid_size": (8, 8),
        "slow_and_differentiable": False,
    }
    test_args = (torch.rand(2, 3, 10, 20),)
    test_kwargs = {
        "clip_limit": 20.0,
        "grid_size": (4, 4),
        "slow_and_differentiable": False,
    }
    _test_function(
        kornia.enhance.equalize_clahe,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
        skip=True,
    )


# TODO: DCF handling: uses a loop to iterate over the batch dim(https://github.com/kornia/kornia/blob/bdeac07e1edc26863a9c8d0826dc202974fd850a/kornia/enhance/adjust.py#L968)
def test_equalize3d(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 2, 3, 3, 3),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 2, 3, 3, 3),)
    test_kwargs = {}
    _test_function(
        kornia.enhance.equalize3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
        skip=True,
    )


def test_histogram(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 10),
        torch.linspace(0, 255, 128),
        torch.tensor(0.9),
    )
    trace_kwargs = {"epsilon": 1e-10}
    test_args = (
        torch.rand(5, 10),
        torch.linspace(0, 255, 128),
        torch.tensor(0.9),
    )
    test_kwargs = {"epsilon": 1e-10}
    _test_function(
        kornia.enhance.histogram,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_histogram2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 32),
        torch.rand(2, 32),
        torch.linspace(0, 255, 128),
        torch.tensor(0.9),
    )
    trace_kwargs = {"epsilon": 1e-10}
    test_args = (
        torch.rand(5, 32),
        torch.rand(5, 32),
        torch.linspace(0, 255, 128),
        torch.tensor(0.9),
    )
    test_kwargs = {"epsilon": 1e-10}
    _test_function(
        kornia.enhance.histogram2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_image_histogram2d(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 1, 10, 10),)
    trace_kwargs = {
        "min": 0.0,
        "max": 255.0,
        "n_bins": 256,
        "bandwidth": None,
        "centers": None,
        "return_pdf": False,
        "kernel": "triangular",
        "eps": 1e-10,
    }
    test_args = (torch.rand(5, 1, 10, 10),)
    test_kwargs = {
        "min": 0.0,
        "max": 255.0,
        "n_bins": 256,
        "bandwidth": None,
        "centers": None,
        "return_pdf": False,
        "kernel": "triangular",
        "eps": 1e-10,
    }
    _test_function(
        kornia.enhance.image_histogram2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_normalize(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
        torch.tensor([0.5, 0.5, 0.5]),
        torch.tensor([0.5, 0.5, 0.5]),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 4),
        torch.tensor([0.4, 0.4, 0.4]),
        torch.tensor([0.6, 0.6, 0.6]),
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.normalize,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_normalize_min_max(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 3, 4, 4), 0.0, 1.0)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3, 4, 4), -1.0, 1.0)
    test_kwargs = {}
    _test_function(
        kornia.enhance.normalize_min_max,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_denormalize(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
        torch.tensor([0.5, 0.5, 0.5]),
        torch.tensor([0.5, 0.5, 0.5]),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 4),
        torch.tensor([0.4, 0.4, 0.4]),
        torch.tensor([0.6, 0.6, 0.6]),
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.denormalize,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


# NOTE: numerical instability in svd() leads to logits not being allclose
def test_zca_mean(target_framework, mode, backend_compile):
    trace_args = (torch.rand(10, 20),)
    trace_kwargs = {"dim": 0}
    test_args = (torch.rand(5, 10, 20),)
    test_kwargs = {"dim": 0}
    _test_function(
        kornia.enhance.zca_mean,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1000,
        mode=mode,
    )


# NOTE: numerical instability in svd() leads to logits not being allclose
def test_zca_whiten(target_framework, mode, backend_compile):
    trace_args = (torch.rand(10, 20),)
    trace_kwargs = {"dim": 0}
    test_args = (torch.rand(5, 10, 20),)
    test_kwargs = {"dim": 0}
    _test_function(
        kornia.enhance.zca_whiten,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1000,
        mode=mode,
    )


def test_linear_transform(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn((5, 3, 4, 5)),
        torch.ones((5 * 3 * 4, 5 * 3 * 4)),
        2 * torch.ones((1, 5 * 3 * 4)),
    )
    trace_kwargs = {"dim": 0}
    test_args = (
        torch.randn((10, 3, 4, 5)),
        torch.ones((10 * 3 * 4, 10 * 3 * 4)),
        2 * torch.ones((1, 10 * 3 * 4)),
    )
    test_kwargs = {"dim": 3}
    _test_function(
        kornia.enhance.linear_transform,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_jpeg_codec_differentiable(target_framework, mode, backend_compile):
    trace_args = (torch.rand(3, 3, 64, 64), torch.tensor([99.0]))
    trace_kwargs = {}
    test_args = (torch.rand(5, 3, 3, 64, 64), torch.tensor([50.0]))
    test_kwargs = {}
    _test_function(
        kornia.enhance.jpeg_codec_differentiable,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_Normalize(target_framework, mode, backend_compile):
    print("kornia.enhance.Normalize")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    torch_init_args = (
        torch.zeros(4),
        255. * torch.rand(4),
    )
    x = torch.rand(1, 4, 3, 3)
    torch_out = kornia.enhance.Normalize(*torch_init_args)(x)

    transpiled_init_args = _nest_torch_tensor_to_new_framework(torch_init_args, target_framework)
    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = transpiled_kornia.enhance.Normalize(*transpiled_init_args)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_Denormalize(target_framework, mode, backend_compile):
    print("kornia.enhance.Denormalize")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    torch_init_args = (
        torch.zeros(1, 4),
        255. * torch.rand(1, 4),
    )
    x = torch.rand(1, 4, 3, 3, 3)
    torch_out = kornia.enhance.Denormalize(*torch_init_args)(x)

    transpiled_init_args = _nest_torch_tensor_to_new_framework(torch_init_args, target_framework)
    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = transpiled_kornia.enhance.Denormalize(*transpiled_init_args)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_ZCAWhitening(target_framework, mode, backend_compile):
    print("kornia.enhance.ZCAWhitening")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    x = torch.tensor([[0,1],[1,0],[-1,0],[0,-1]], dtype = torch.float32)
    zca = kornia.enhance.ZCAWhitening().fit(x)
    torch_out = zca(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_zca = transpiled_kornia.enhance.ZCAWhitening().fit(transpiled_x)
    transpiled_out = transpiled_zca(x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_AdjustBrightness(target_framework, mode, backend_compile):
    print("kornia.enhance.AdjustBrightness")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    x = torch.rand(2, 5, 3, 3)
    y = torch.rand(2)
    torch_out = kornia.enhance.AdjustBrightness(y)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_y = _nest_torch_tensor_to_new_framework(y, target_framework)
    transpiled_out = transpiled_kornia.enhance.AdjustBrightness(transpiled_y)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_AdjustContrast(target_framework, mode, backend_compile):
    print("kornia.enhance.AdjustContrast")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    x = torch.rand(2, 5, 3, 3)
    y = torch.rand(2)
    torch_out = kornia.enhance.AdjustContrast(y)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_y = _nest_torch_tensor_to_new_framework(y, target_framework)
    transpiled_out = transpiled_kornia.enhance.AdjustContrast(transpiled_y)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_AdjustSaturation(target_framework, mode, backend_compile):
    print("kornia.enhance.AdjustSaturation")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    x = torch.rand(2, 3, 3, 3)
    y = torch.rand(2)
    torch_out = kornia.enhance.AdjustSaturation(y)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_y = _nest_torch_tensor_to_new_framework(y, target_framework)
    transpiled_out = transpiled_kornia.enhance.AdjustSaturation(transpiled_y)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_AdjustHue(target_framework, mode, backend_compile):
    print("kornia.enhance.AdjustHue")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    x = torch.rand(2, 3, 3, 3)
    y = torch.ones(2) * 3.141516
    torch_out = kornia.enhance.AdjustHue(y)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_y = _nest_torch_tensor_to_new_framework(y, target_framework)
    transpiled_out = transpiled_kornia.enhance.AdjustHue(transpiled_y)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_AdjustGamma(target_framework, mode, backend_compile):
    print("kornia.enhance.AdjustGamma")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    x = torch.rand(2, 5, 3, 3)
    torch_init_args = (
        torch.ones(2) * 1.0,
        torch.ones(2) * 2.0,
    )
    torch_out = kornia.enhance.AdjustGamma(*torch_init_args)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_init_args = _nest_torch_tensor_to_new_framework(torch_init_args, target_framework)
    transpiled_out = transpiled_kornia.enhance.AdjustGamma(*transpiled_init_args)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_AdjustSigmoid(target_framework, mode, backend_compile):
    print("kornia.enhance.AdjustSigmoid")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    x = torch.rand(1, 1, 2, 2)
    torch_out = kornia.enhance.AdjustSigmoid(gain=0)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = transpiled_kornia.enhance.AdjustSigmoid(gain=0)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_AdjustLog(target_framework, mode, backend_compile):
    print("kornia.enhance.AdjustLog")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    x = torch.rand(1, 1, 2, 2)
    torch_out = kornia.enhance.AdjustLog(inv=True)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = transpiled_kornia.enhance.AdjustLog(inv=True)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_AddWeighted(target_framework, mode, backend_compile):
    print("kornia.enhance.AddWeighted")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    init_args = (0.5, 0.5, 1.0)
    torch_call_args = (
        torch.rand(1, 1, 5, 5),
        torch.rand(1, 1, 5, 5),
    )
    torch_out = kornia.enhance.AddWeighted(*init_args)(*torch_call_args)

    transpiled_call_args = _nest_torch_tensor_to_new_framework(torch_call_args, target_framework)
    transpiled_out = transpiled_kornia.enhance.AddWeighted(*init_args)(*transpiled_call_args)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_Invert(target_framework, mode, backend_compile):
    print("kornia.enhance.Invert")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    x = torch.rand(1, 2, 4, 4)
    torch_out = kornia.enhance.Invert()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = transpiled_kornia.enhance.Invert()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_JPEGCodecDifferentiable(target_framework, mode, backend_compile):
    print("kornia.enhance.JPEGCodecDifferentiable")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    torch_args = (
        torch.rand(2, 3, 32, 32, dtype=torch.float),
        torch.tensor((99.0, 1.0)),
    )
    torch_out = kornia.enhance.JPEGCodecDifferentiable()(*torch_args)

    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)
    transpiled_out = transpiled_kornia.enhance.JPEGCodecDifferentiable()(*transpiled_args)

    _to_numpy_and_allclose(torch_out, transpiled_out)
