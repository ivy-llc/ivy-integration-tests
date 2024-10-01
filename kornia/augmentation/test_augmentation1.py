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
    if backend_compile or target == "numpy":
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

def test_ColorJiggle(target_framework, mode, backend_compile):
    print("kornia.augmentation.ColorJiggle")
    
    init_args = (0.1, 0.1, 0.1, 0.1)
    init_kwargs = {"p": 1.}
    call_args = (torch.ones(1, 3, 3, 3),)
    call_kwargs = {}
    
    # TODO: test previous parameter state

    _test_augmentation_class(
        kornia.augmentation.ColorJiggle,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_ColorJitter(target_framework, mode, backend_compile):
    print("kornia.augmentation.ColorJitter")

    init_args = (0.1, 0.1, 0.1, 0.1)
    init_kwargs = {"p": 1.}
    call_args = (torch.ones(1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.ColorJitter,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomAutoContrast(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomAutoContrast")

    init_args = ()
    init_kwargs = {}
    call_args = (torch.ones(5, 3, 4, 4),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomAutoContrast,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomBoxBlur(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomBoxBlur")

    init_args = ((7, 7),)
    init_kwargs = {}
    call_args = (torch.ones(1, 1, 24, 24),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomBoxBlur,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomBrightness(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomBrightness")

    init_args = ()
    init_kwargs = {"brightness": (0.5, 2.), "p": 1.}
    call_args = (torch.rand(1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomBrightness,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomChannelDropout(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomChannelDropout")

    init_args = ()
    init_kwargs = {"num_drop_channels": 1, "fill_value": 0.0, "p": 1.0}
    call_args = (torch.ones(1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomChannelDropout,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomChannelShuffle(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomChannelShuffle")

    init_args = ()
    init_kwargs = {}
    call_args = (torch.arange(1*2*2*2.).view(1,2,2,2),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomChannelShuffle,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomClahe(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomClahe")

    init_args = ()
    init_kwargs = {}
    call_args = (torch.rand(2, 3, 10, 20),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomClahe,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomContrast(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomContrast")

    init_args = ()
    init_kwargs = {"contrast": (0.5, 2.), "p": 1.}
    call_args = (torch.rand(1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomContrast,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomEqualize(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomEqualize")

    init_args = ()
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 1, 5, 5),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomEqualize,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomGamma(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomGamma")

    init_args = ((0.5, 2.), (1.5, 1.5))
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomGamma,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomGaussianBlur(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomGaussianBlur")

    init_args = ((3, 3), (0.1, 2.0))
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 1, 5, 5),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomGaussianBlur,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomGaussianIllumination(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomGaussianIllumination")

    init_args = ()
    init_kwargs = {"gain": 0.5, "p": 1.}
    call_args = (torch.ones(1, 3, 3, 3) * 0.5,)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomGaussianIllumination,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomGaussianNoise(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomGaussianNoise")

    init_args = ()
    init_kwargs = {"mean": 0., "std": 1., "p": 1.}
    call_args = (torch.ones(1, 1, 2, 2),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomGaussianNoise,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomGrayscale(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomGrayscale")

    init_args = ()
    init_kwargs = {"p": 1.}
    call_args = (torch.randn((1, 3, 3, 3)),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomGrayscale,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomHue(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomHue")

    init_args = ()
    init_kwargs = {"hue": (-0.5, 0.5), "p": 1.}
    call_args = (torch.rand(1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomHue,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomInvert(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomInvert")

    init_args = ()
    init_kwargs = {}
    call_args = (torch.rand(1, 1, 5, 5),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomInvert,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomJPEG(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomJPEG")

    init_args = ()
    init_kwargs = {"jpeg_quality": (1.0, 50.0), "p": 1.}
    call_args = (0.1904 * torch.ones(2, 3, 32, 32), )
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomJPEG,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )
