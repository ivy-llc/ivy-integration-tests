from helpers import (
    _check_allclose,
    _check_shape_allclose,
    _nest_array_to_numpy,
    _nest_torch_tensor_to_new_framework,
)

import ivy
import kornia
import pytest
import torch


# Helpers #
# ------- #

def _test_augmentation_class(
    augmentation_cls,
    target,
    init_args=(),
    init_kwargs={},
    call_args=(),
    call_kwargs={},
    deterministic_output=True,
    backend_compile=False,
    tolerance=1e-3,
):
    if backend_compile:
        pytest.skip()

    transpiled_cls = ivy.transpile(augmentation_cls, source="torch", target=target)

    torch_aug = augmentation_cls(*init_args, **init_kwargs)
    transpiled_init_args = _nest_torch_tensor_to_new_framework(init_args, target)
    transpiled_init_kwargs = _nest_torch_tensor_to_new_framework(init_kwargs, target)
    transpiled_aug = transpiled_cls(*transpiled_init_args, **transpiled_init_kwargs)

    # assert dir(torch_aug) == dir(transpiled_aug), f"attributes/methods of transpiled object do not align with the original - orig: {dir(torch_aug)} != transpiled: {dir(transpiled_aug)}"

    torch_out = torch_aug(*call_args, **call_kwargs)
    transpiled_call_args = _nest_torch_tensor_to_new_framework(call_args, target)
    transpiled_call_kwargs = _nest_torch_tensor_to_new_framework(call_kwargs, target)
    transpiled_out = transpiled_aug(*transpiled_call_args, **transpiled_call_kwargs)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)

    _check_shape_allclose(orig_np, transpiled_np)

    if deterministic_output:
        orig_np = _nest_array_to_numpy(torch_out)
        transpiled_np = _nest_array_to_numpy(transpiled_out)
        _check_allclose(orig_np, transpiled_np, tolerance=tolerance)
    else:
        # TODO: add value test to ensure the output of 
        # `transpiled_aug(*transpiled_call_args, **transpiled_call_kwargs)`
        # changes most times it runs ??
        # or verify the mean and std are within a given range??
        pass


# Tests #
# ----- #

def test_RandomLinearCornerIllumination(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomLinearCornerIllumination")

    init_args = ()
    init_kwargs = {"gain": 0.25, "p": 1.}
    call_args = (torch.ones(1, 3, 3, 3) * 0.5,)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomLinearCornerIllumination,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomLinearIllumination(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomLinearIllumination")

    init_args = ()
    init_kwargs = {"gain": 0.25, "p": 1.}
    call_args = (torch.ones(1, 3, 3, 3) * 0.5,)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomLinearIllumination,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomMedianBlur(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomMedianBlur")

    init_args = ((3, 3),)
    init_kwargs = {"p": 1}
    call_args = (torch.ones(1, 1, 4, 4),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomMedianBlur,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomMotionBlur(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomMotionBlur")

    init_args = (3, 35., 0.5)
    init_kwargs = {"p": 1.}
    call_args = (torch.ones(1, 1, 5, 5),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomMotionBlur,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomPlanckianJitter(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomPlanckianJitter")

    init_args = ()
    init_kwargs = {"mode": "blackbody", "select_from": [23, 24, 1, 2]}
    call_args = (torch.randn(2, 3, 2, 2),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomPlanckianJitter,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomPlasmaBrightness(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomPlasmaBrightness")

    init_args = ()
    init_kwargs = {"roughness": (0.1, 0.7), "p": 1.}
    call_args = (torch.ones(1, 1, 3, 4),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomPlasmaBrightness,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomPlasmaContrast(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomPlasmaContrast")

    init_args = ()
    init_kwargs = {"roughness": (0.1, 0.7), "p": 1.}
    call_args = (torch.ones(1, 1, 3, 4),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomPlasmaContrast,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomPlasmaShadow(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomPlasmaShadow")

    init_args = ()
    init_kwargs = {"roughness": (0.1, 0.7), "p": 1.}
    call_args = (torch.ones(1, 1, 3, 4),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomPlasmaShadow,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomPosterize(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomPosterize")

    init_args = (3.,)
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 1, 5, 5),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomPosterize,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomRain(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomRain")

    init_args = ()
    init_kwargs = {"p": 1, "drop_height": (1,2), "drop_width": (1,2), "number_of_drops": (1,1)}
    call_args = (torch.rand(1, 1, 5, 5),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomRain,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomRGBShift(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomRGBShift")

    init_args = ()
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 3, 5, 5),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomRGBShift,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomSaltAndPepperNoise(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomSaltAndPepperNoise")

    init_args = ()
    init_kwargs = {"amount": 0.5, "salt_vs_pepper": 0.5, "p": 1.}
    call_args = (torch.rand(1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomSaltAndPepperNoise,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomSaturation(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomSaturation")

    init_args = ()
    init_kwargs = {"saturation": (0.5, 2.), "p": 1.}
    call_args = (torch.rand(1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomSaturation,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomSharpness(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomSharpness")

    init_args = (1.,)
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 1, 5, 5),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomSharpness,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomSnow(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomSnow")

    init_args = ()
    init_kwargs = {"p": 1.0, "snow_coefficient": (0.1, 0.6), "brightness": (1.0, 5.0)}
    call_args = (torch.rand(2, 3, 4, 4),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomSnow,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomSolarize(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomSolarize")

    init_args = (0.1, 0.1)
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 1, 5, 5),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomSolarize,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_CenterCrop(target_framework, mode, backend_compile):
    print("kornia.augmentation.CenterCrop")

    init_args = (2,)
    init_kwargs = {"p": 1., "cropping_mode": "resample"}
    call_args = (torch.randn(1, 1, 4, 4),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.CenterCrop,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )
    
    # TODO: test CenterCrop.inverse
