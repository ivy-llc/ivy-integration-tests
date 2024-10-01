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

def test_RandomMosaic(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomMosaic")

    init_args = ((300, 300),)
    init_kwargs = {"data_keys": ["input", "bbox_xyxy"]}
    call_args = (
        torch.randn(8, 3, 224, 224),
        torch.tensor([[
            [70, 5, 150, 100],
            [60, 180, 175, 220],
        ]]).repeat(8, 1, 1),
    )
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomMosaic,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomTransplantation(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomTransplantation")

    init_args = ()
    init_kwargs = {"p": 1.}
    call_args = (torch.randn(2, 3, 5, 5), torch.randint(0, 3, (2, 5, 5)))
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomTransplantation,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_CenterCrop3D(target_framework, mode, backend_compile):
    print("kornia.augmentation.CenterCrop3D")

    init_args = (2,)
    init_kwargs = {"p": 1.}
    call_args = (torch.randn(1, 1, 2, 4, 6),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.CenterCrop3D,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomAffine3D(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomAffine3D")

    init_args = ((15., 20., 20.),)
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomAffine3D,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomCrop3D(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomCrop3D")

    init_args = ((2, 2, 2),)
    init_kwargs = {"p": 1.}
    call_args = (torch.randn(1, 1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomCrop3D,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomDepthicalFlip3D(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomDepthicalFlip3D")

    init_args = ()
    init_kwargs = {"p": 1.}
    call_args = (torch.eye(3).repeat(3, 1, 1),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomDepthicalFlip3D,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomHorizontalFlip3D(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomHorizontalFlip3D")

    init_args = ()
    init_kwargs = {"p": 1.}
    call_args = (torch.eye(3).repeat(3, 1, 1),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomHorizontalFlip3D,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomRotation3D(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomRotation3D")

    init_args = ((15., 20., 20.),)
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomRotation3D,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomVerticalFlip3D(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomVerticalFlip3D")

    init_args = ()
    init_kwargs = {"p": 1.}
    call_args = (torch.eye(3).repeat(3, 1, 1),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomVerticalFlip3D,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomEqualize3D(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomEqualize3D")

    init_args = ()
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomEqualize3D,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomMotionBlur3D(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomMotionBlur3D")

    init_args = (3, 35., 0.5)
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 1, 3, 5, 5),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomMotionBlur3D,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomTransplantation3D(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomTransplantation3D")

    init_args = ()
    init_kwargs = {"p": 1.}
    call_args = (torch.randn(2, 3, 5, 5), torch.randint(0, 3, (2, 5, 5)))
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomTransplantation3D,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_Denormalize(target_framework, mode, backend_compile):
    print("kornia.augmentation.Denormalize")

    init_args = ()
    init_kwargs = {"mean": torch.zeros(1, 4), "std": torch.ones(1, 4)}
    call_args = (torch.rand(1, 4, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.Denormalize,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_Normalize(target_framework, mode, backend_compile):
    print("kornia.augmentation.Normalize")

    init_args = ()
    init_kwargs = {"mean": torch.zeros(4), "std": torch.ones(4)}
    call_args = (torch.rand(1, 4, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.Normalize,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_LongestMaxSize(target_framework, mode, backend_compile):
    print("kornia.augmentation.LongestMaxSize")

    init_args = (100,)
    init_kwargs = {}
    call_args = (torch.rand(10, 3, 200, 200),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.LongestMaxSize,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=True,
        backend_compile=backend_compile,
    )


def test_Resize(target_framework, mode, backend_compile):
    print("kornia.augmentation.Resize")

    init_args = ((100, 100),)
    init_kwargs = {}
    call_args = (torch.rand(10, 3, 50, 50),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.Resize,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=True,
        backend_compile=backend_compile,
    )


def test_SmallestMaxSize(target_framework, mode, backend_compile):
    print("kornia.augmentation.SmallestMaxSize")

    init_args = (100,)
    init_kwargs = {}
    call_args = (torch.rand(10, 3, 50, 50),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.SmallestMaxSize,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=True,
        backend_compile=backend_compile,
    )
