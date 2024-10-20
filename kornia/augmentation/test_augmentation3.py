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

    transpiled_kornia = ivy.transpile(kornia, target=target)
    transpiled_cls = eval("transpiled_" + f"{augmentation_cls.__module__}.{augmentation_cls.__name__}")

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

def test_PadTo(target_framework, mode, backend_compile):
    print("kornia.augmentation.PadTo")

    init_args = ((4, 5),)
    init_kwargs = {"pad_value": 1.}
    call_args = (torch.tensor([[[[0., 0., 0.],
                                 [0., 0., 0.],
                                 [0., 0., 0.]]]]),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.PadTo,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomAffine(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomAffine")

    init_args = ((-15., 20.),)
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 1, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomAffine,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )

    # TODO: test RandomAffine.inverse, RandomAffine.transform_matrix


def test_RandomCrop(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomCrop")

    init_args = ((2, 2),)
    init_kwargs = {"p": 1., "cropping_mode": "resample"}
    call_args = (torch.arange(1*1*3*3.).view(1, 1, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomCrop,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomElasticTransform(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomElasticTransform")

    init_args = ()
    init_kwargs = {}
    call_args = (torch.ones(1, 1, 2, 2),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomElasticTransform,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomErasing(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomErasing")

    init_args = ((.4, .8), (.3, 1/.3))
    init_kwargs = {"p": 0.5}
    call_args = (torch.ones(1, 1, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomErasing,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomFisheye(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomFisheye")

    init_args = (
        torch.tensor([-.3, .3]),
        torch.tensor([-.3, .3]),
        torch.tensor([.9, 1.]),
    )
    init_kwargs = {}
    call_args = (torch.randn(1, 3, 32, 32),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomFisheye,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomHorizontalFlip(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomHorizontalFlip")

    init_args = ()
    init_kwargs = {"p": 1.}
    call_args = (torch.tensor([[[[0., 0., 0.],
                                 [0., 0., 0.],
                                 [0., 1., 1.]]]]),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomHorizontalFlip,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomPerspective(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomPerspective")

    init_args = (0.5,)
    init_kwargs = {"p": 0.5}
    call_args = (torch.tensor([[[[1., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 1.]]]]),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomPerspective,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomResizedCrop(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomResizedCrop")

    init_args = ()
    init_kwargs = {"size": (3, 3), "scale": (3., 3.), "ratio": (2., 2.), "p": 1., "cropping_mode": "resample"}
    call_args = (torch.tensor([[[0., 1., 2.],
                                [3., 4., 5.],
                                [6., 7., 8.]]]),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomResizedCrop,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomRotation(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomRotation")

    init_args = ()
    init_kwargs = {"degrees": 45.0, "p": 1.}
    call_args = (torch.tensor([[1., 0., 0., 2.],
                               [0., 0., 0., 0.],
                               [0., 1., 2., 0.],
                               [0., 0., 1., 2.]]),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomRotation,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomShear(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomShear")

    init_args = ((-5., 2., 5., 10.),)
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 1, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomShear,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomThinPlateSpline(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomThinPlateSpline")

    init_args = ()
    init_kwargs = {}
    call_args = (torch.randn(1, 3, 32, 32),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomThinPlateSpline,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomVerticalFlip(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomVerticalFlip")

    init_args = ()
    init_kwargs = {"p": 1.}
    call_args = (torch.randn(1, 3, 32, 32),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomVerticalFlip,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomCutMixV2(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomCutMixV2")

    input = torch.rand(2, 1, 3, 3)
    input[0] = torch.ones((1, 3, 3))
    label = torch.tensor([0, 1])

    init_args = ()
    init_kwargs = {"data_keys": ["input", "class"]}
    call_args = (input, label)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomCutMixV2,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomJigsaw(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomJigsaw")

    init_args = ((4, 4),)
    init_kwargs = {}
    call_args = (torch.randn(8, 3, 256, 256),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomJigsaw,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomMixUpV2(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomMixUpV2")

    init_args = ()
    init_kwargs = {"data_keys": ["input", "class"]}
    call_args = (torch.rand(2, 1, 3, 3), torch.tensor([0, 1]))
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomMixUpV2,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )
